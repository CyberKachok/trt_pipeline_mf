import os
import cv2
import time
import argparse
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_config_seqs(matlab_file_path):
    """Парсинг MATLAB конфига"""
    sequences = []

    try:
        with open(matlab_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: MATLAB config file not found at {matlab_file_path}")
        return sequences

    seq_pattern = re.compile(
        r"struct\('name','([^']*)','path','([^']*)','startFrame',(\d+),'endFrame',(\d+),'nz',(\d+),'ext','([^']*)','init_rect',\s*\[([\d.,\s]+)\]\),?",
        re.DOTALL
    )

    uav123_block_match = re.search(r"seqUAV123\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if uav123_block_match:
        uav123_block = uav123_block_match.group(1)

        for match in seq_pattern.finditer(uav123_block):

            path = match.group(2).replace('\\', '/')
            path = path.rstrip('/')
            folder_name = os.path.basename(path)

            init_rect_str = match.group(7)
            init_rect = list(map(float, init_rect_str.replace(' ', '').split(',')))

            sequences.append({
                'name': match.group(1),
                'folder_path': folder_name,
                'startFrame': int(match.group(3)),
                'endFrame': int(match.group(4)),
                'nz': int(match.group(5)),
                'ext': match.group(6),
                'init_rect': init_rect
            })

    if not sequences:
        print("Warning: No sequences found in the MATLAB config file.")

    return sequences


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Ensure inputs are numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Check for NaN values
    if np.isnan(box1).any() or np.isnan(box2).any():
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Handle invalid boxes with non-positive width or height
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area < 1e-6:
        return 0.0

    return float(intersection_area / union_area)


def calculate_center_error(box1, box2):
    """Calculate center location error between two bounding boxes."""
    # Ensure inputs are numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)

    # box format: [x, y, w, h]
    center1_x = box1[0] + box1[2] / 2
    center1_y = box1[1] + box1[3] / 2
    center2_x = box2[0] + box2[2] / 2
    center2_y = box2[1] + box2[3] / 2

    if any(np.isnan(v) for v in [center1_x, center1_y, center2_x, center2_y]):
        return 100.0  # Large error value

    return float(np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2))


def calculate_diagonal(box):
    """Calculate the diagonal length of a bounding box."""
    # box format: [x, y, w, h]
    return np.sqrt(box[2] ** 2 + box[3] ** 2)


def calculate_precision_at_thresholds(center_errors, thresholds):
    """Calculate precision at various pixel distance thresholds."""
    center_errors_np = np.array(center_errors)
    precision = []
    for t in thresholds:
        precision.append(np.mean(center_errors_np <= t))
    return np.array(precision)


def calculate_success_rate_at_thresholds(ious, thresholds):
    """Calculate success rate at various IoU thresholds."""
    ious_np = np.array(ious)
    success_rate = []
    for t in thresholds:
        success_rate.append(np.mean(ious_np > t))
    return np.array(success_rate)


def load_ground_truth_boxes(gt_file):
    """Load ground truth bounding boxes from annotation file."""
    if not os.path.exists(gt_file):
        return None

    try:
        with open(gt_file, 'r') as f:
            gt_boxes = []
            for line in f:
                line = line.strip()
                if line:
                    if ',' in line:
                        coords = list(map(float, line.split(',')))
                    else:
                        coords = list(map(float, line.split()))
                    gt_boxes.append(coords)
        return gt_boxes
    except Exception as e:
        print(f"Error loading ground truth file {gt_file}: {e}")
        return None


def parse_attribute_file(attribute_file_path):
    """Парсинг файлов аттрибутов для каждой отдельной последовательности"""
    attributes = {}
    if not os.path.exists(attribute_file_path):
        # print(f"Att file not found at {attribute_file_path}")
        return attributes

    try:
        with open(attribute_file_path, 'r') as f:
            content = f.read().strip()
            if content:
                attr_values = [int(x) for x in re.split(r'[,\s]+', content) if x]

                attribute_names = ['IV', 'SV', 'POC', 'FOC', 'OV', 'FM', 'CM', 'BC', 'SOB', 'ARC', 'VC', 'LR'] # TODO: Check

                for i, attr_name in enumerate(attribute_names):
                    if i < len(attr_values):
                        attributes[attr_name] = attr_values[i]
                    else:
                        attributes[attr_name] = 0
    except Exception as e:
        print(f"Error parsing attribute file {attribute_file_path}: {e}")
    return attributes


def parse_all_attributes(anno_root, uav123_sequences):
    """Парсинг файлов аттрибутов"""
    all_sequences_attributes = {}
    att_dir = os.path.join(anno_root, 'att')

    if not os.path.exists(att_dir):
        print(f"Warning: Attribute directory not found at {att_dir}. Skipping attribute parsing.")
        return all_sequences_attributes

    for seq_info in uav123_sequences:
        seq_name = seq_info['name']
        attribute_file_path = os.path.join(att_dir, f"{seq_name}.txt")
        attributes = parse_attribute_file(attribute_file_path)
        if attributes:
            all_sequences_attributes[seq_name] = attributes

    if not all_sequences_attributes:
        print("Warning: No attributes loaded from individual files. Attribute-based metrics will be skipped.")

    return all_sequences_attributes


def build_image_file_list(seq_info, data_root):
    """Build list of image file paths for a sequence."""
    seq_path = os.path.join(data_root, seq_info['folder_path'])

    if not os.path.exists(seq_path):
        print(f"Warning: Sequence directory not found: {seq_path}")
        print(f"  Sequence: {seq_info['name']}")
        print(f"  Looking for folder: {seq_info['folder_path']}")
        return []

    img_files = []

    for i in range(seq_info['startFrame'], seq_info['endFrame'] + 1):
        img_filename = f"{i:0{seq_info['nz']}d}.{seq_info['ext']}"
        img_files.append(os.path.join(seq_path, img_filename))

    return img_files


def initialize_tracker(ckpt_path, cfg_path):
    """Initialize the appropriate tracker based on checkpoint format."""
    if ckpt_path.endswith(('.pth', '.pt')):
        from model.torch_tracker_wrapper import TorchTrackerWrapper
        #from model.torch_tracker_wrapper import TorchTrackerWrapper
        return TorchTrackerWrapper(cfg_path, ckpt_path)
    elif ckpt_path.endswith(('.engine', '.trt')):
        from model.trt_tracker_wrapper import TrtTrackerWrapper
        return TrtTrackerWrapper(cfg_path, ckpt_path)
    else:
        raise ValueError(f"Unsupported model format: {ckpt_path}")

def tensor_to_float(tensor_or_value):
    value = float(tensor_or_value)
    return 0.0 if np.isnan(value) else value


def process_sequence(seq_info, cfg_path, ckpt_path, data_root, anno_root, visualize=False):
    """Проход по одной последовательности"""
    seq_name = seq_info['name']

    img_files = build_image_file_list(seq_info, data_root)
    if not img_files:
        print(f"Warning: No image files found for sequence {seq_name}. Skipping.")
        return None

    gt_file = os.path.join(anno_root, f"{seq_name}.txt")
    gt_boxes = load_ground_truth_boxes(gt_file)
    if gt_boxes is None or len(gt_boxes) == 0:
        print(f"Warning: No valid ground truth for {seq_name}. Skipping sequence.")
        return None

    if not os.path.exists(img_files[0]):
        print(f"Warning: First frame not found for {seq_name}: {img_files[0]}. Skipping sequence.")
        return None


    try:
        tracker = initialize_tracker(ckpt_path, cfg_path)

        init_frame = cv2.cvtColor(cv2.imread(img_files[0]), cv2.COLOR_BGR2RGB)
        init_gt_box = gt_boxes[0]
        init_bbox_xyxy = [
            init_gt_box[0],
            init_gt_box[1],
            init_gt_box[0] + init_gt_box[2],
            init_gt_box[1] + init_gt_box[3]
        ]

        tracker.initialize(init_frame, init_bbox_xyxy)

        iou_list = []
        center_error_list = []
        normalized_center_error_list = []
        time_list = []

        for idx, img_path in enumerate(tqdm(img_files, desc=seq_name, leave=False)):
            if not os.path.exists(img_path):
                print(f"Warning: Frame {img_path} not found. Skipping.")
                continue

            frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            start_time = time.time()
            tracking_result = tracker.track(frame, idx + 1)
            #time.sleep(0.3)
            tracking_time = time.time() - start_time
            time_list.append(tracking_time)

            bbox_result = tracking_result[0]

            x1 = tensor_to_float(bbox_result[0])
            y1 = tensor_to_float(bbox_result[1])
            x2 = tensor_to_float(bbox_result[2])
            y2 = tensor_to_float(bbox_result[3])

            pred_bbox_xyxy = [x1, y1, x2, y2]

            if idx < len(gt_boxes):
                gt_box = gt_boxes[idx]

                pred_bbox_wh = [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ]

                iou = calculate_iou(pred_bbox_wh, gt_box)
                center_error = calculate_center_error(pred_bbox_wh, gt_box)

                gt_diagonal = calculate_diagonal(gt_box)
                normalized_center_error = center_error / gt_diagonal if gt_diagonal > 0 else 100.0

                if np.isnan(iou):
                    iou = 0.0
                if np.isnan(center_error):
                    center_error = 100.0
                if np.isnan(normalized_center_error):
                    normalized_center_error = 100.0

                iou_list.append(float(iou))
                center_error_list.append(float(center_error))
                normalized_center_error_list.append(float(normalized_center_error))

            if visualize:
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis_frame,
                              (int(pred_bbox_xyxy[0]), int(pred_bbox_xyxy[1])),
                              (int(pred_bbox_xyxy[2]), int(pred_bbox_xyxy[3])),
                              (0, 0, 255), 2)

                text = f"Score: {tracking_result[1]:.2f}"  # Форматируем текст
                text_position = (int(pred_bbox_xyxy[0]), int(pred_bbox_xyxy[1]) - 10)  # Над верхней границей bbox
                cv2.putText(
                    vis_frame,
                    text,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,  # Шрифт
                    0.7,  # Масштаб
                    (0, 0, 255),  # Цвет (зеленый)
                    2,  # Толщина
                    cv2.LINE_AA  # Сглаживание
                )


                vis_frame = cv2.resize(vis_frame, (920, 640))
                cv2.imshow("Tracking", vis_frame)
                # if seq_name not in  ['car6_4','person13', 'person16', 'person18', 'uav3', 'uav6']:# 'car9', ,
                #     break
                # cv2.waitKey(7)

                if cv2.waitKey(1) == 27:
                    break


        # Calculate summary metrics
        if iou_list and time_list:
            clean_iou_list = [0.0 if np.isnan(iou) else iou for iou in iou_list]
            avg_iou = np.mean(clean_iou_list)
            avg_fps = len(time_list) / sum(time_list)
            avg_np = np.mean(normalized_center_error_list)
            avg_p =  np.mean(center_error_list)

            # Calculate success rates and precision curves
            iou_thresholds = np.arange(0, 1.05, 0.05)
            success_rates = calculate_success_rate_at_thresholds(iou_list, iou_thresholds)
            auc_iou = np.trapz(success_rates, iou_thresholds)

            precision_thresholds = np.arange(0, 51, 1)
            precision_values = calculate_precision_at_thresholds(center_error_list, precision_thresholds)
            auc_precision = np.trapz(precision_values, precision_thresholds)

            failure_rate = np.mean(np.array(iou_list) == 0)

            return {
                'seq_name': seq_name,
                'IoU_list': iou_list,
                'Center_Error_list': center_error_list,
                'Normalized_Center_Error_list': normalized_center_error_list,
                'Avg_IoU': avg_iou,
                'Avg_FPS': avg_fps,
                'Avg_NP': avg_np,
                'Avg_P': avg_p ,
                'AUC_IoU': auc_iou,
                'AUC_Precision': auc_precision,
                'Failure_Rate': failure_rate,
                'IoU_Thresholds': iou_thresholds,
                'Precision_Thresholds': precision_thresholds,
                'Success_Rates_at_Thresholds': success_rates,
                'Precision_at_Thresholds': precision_values
            }

    except Exception as e:
        import traceback
        print(f"Error processing sequence {seq_name}:")
        print(traceback.format_exc())
        return None
    return None


def run_tracking_evaluation(cfg_path, ckpt_path, data_root, anno_root, output_dir,
                            matlab_config_path, annotation_txt_path, visualize=False, debug=False):
    """Основная функция для запуска трекера"""
    print("Starting UAV123 tracking evaluation...")


    # Parse sequences from MATLAB config
    uav123_sequences = parse_config_seqs(matlab_config_path)
    if not uav123_sequences:
        print("Error: No sequences found in configuration file.")
        return None, None, None  # Return None for all three expected values

    print(f"Found {len(uav123_sequences)} sequences to evaluate.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_sequence_results = {}

    if visualize:
        # Sequential processing for visualization
        for seq_info in tqdm(uav123_sequences, desc="Processing sequences"):
            result = process_sequence(seq_info, cfg_path, ckpt_path, data_root, anno_root, visualize)
            if result:
                all_sequence_results[result['seq_name']] = result
    else:
        # Parallel processing with optimal thread count (CPU count or 8, whichever is smaller)
        max_workers = min(8, os.cpu_count() or 4)
        print(f"Processing sequences in parallel with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_sequence, seq_info, cfg_path, ckpt_path,
                                data_root, anno_root, visualize): seq_info['name']
                for seq_info in uav123_sequences
            }

            for future in tqdm(futures, desc="Processing sequences in parallel"):
                seq_name = futures[future]
                result = future.result()
                if result:
                    all_sequence_results[seq_name] = result

    all_sequence_metrics = {}
    for seq_name, results in all_sequence_results.items():
        all_sequence_metrics[seq_name] = {
            'Avg_IoU': results['Avg_IoU'],
            'Avg_FPS': results['Avg_FPS'],
            'Avg_NP': results['Avg_NP'],  # New
            'Avg_P': results['Avg_P'],
            'AUC_IoU': results['AUC_IoU'],
            'AUC_Precision': results['AUC_Precision'],
            'Failure_Rate': results['Failure_Rate']
        }
        print(seq_name, all_sequence_metrics[seq_name]['Avg_IoU'])

    # Parse attributes from individual attribute files in 'att' folder
    uav123_attributes = parse_all_attributes(anno_root, uav123_sequences)

    if not uav123_attributes:
        print("Warning: No attributes loaded. Attribute-based metrics will not be calculated or plotted.")

    # Calculate overall average metrics
    overall_metrics = {
        'Avg_IoU': np.mean([m['Avg_IoU'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,
        'Avg_FPS': np.mean([m['Avg_FPS'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,
        'Avg_NP': np.mean([m['Avg_NP'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,  # New
        'Avg_P': np.mean([m['Avg_P'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,  # New
        'AUC_IoU': np.mean([m['AUC_IoU'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,
        'AUC_Precision': np.mean(
            [m['AUC_Precision'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,
        'Failure_Rate': np.mean(
            [m['Failure_Rate'] for m in all_sequence_metrics.values()]) if all_sequence_metrics else 0,
    }

    # Calculate attribute-based metrics
    # Initialize attribute_metrics with all possible attribute names found
    all_attribute_names = set()
    for attrs in uav123_attributes.values():
        all_attribute_names.update(attrs.keys())

    attribute_metrics = {
        attr: {'Avg_IoU': [], 'Avg_FPS': [], 'Avg_NP': [],  'Avg_P': [], 'AUC_IoU': [], 'AUC_Precision': [], 'Failure_Rate': []}
        for attr in all_attribute_names}

    for seq_name, metrics in all_sequence_metrics.items():
        if seq_name in uav123_attributes:
            for attr, value in uav123_attributes[seq_name].items():
                if value == 1:  # If the attribute is present for this sequence
                    attribute_metrics[attr]['Avg_IoU'].append(metrics['Avg_IoU'])
                    attribute_metrics[attr]['Avg_FPS'].append(metrics['Avg_FPS'])
                    attribute_metrics[attr]['Avg_NP'].append(metrics['Avg_NP'])
                    attribute_metrics[attr]['Avg_P'].append(metrics['Avg_P'])
                    attribute_metrics[attr]['AUC_IoU'].append(metrics['AUC_IoU'])
                    attribute_metrics[attr]['AUC_Precision'].append(metrics['AUC_Precision'])
                    attribute_metrics[attr]['Failure_Rate'].append(metrics['Failure_Rate'])

    # Average attribute metrics
    averaged_attribute_metrics = {}
    for attr, metrics_list in attribute_metrics.items():
        averaged_attribute_metrics[attr] = {
            'Avg_IoU': np.mean(metrics_list['Avg_IoU']) if metrics_list['Avg_IoU'] else 0,
            'Avg_FPS': np.mean(metrics_list['Avg_FPS']) if metrics_list['Avg_FPS'] else 0,
            'Avg_NP': np.mean(metrics_list['Avg_NP']) if metrics_list['Avg_NP'] else 0,
            'Avg_P': np.mean(metrics_list['Avg_P']) if metrics_list['Avg_P'] else 0,
            'AUC_IoU': np.mean(metrics_list['AUC_IoU']) if metrics_list['AUC_IoU'] else 0,
            'AUC_Precision': np.mean(metrics_list['AUC_Precision']) if metrics_list['AUC_Precision'] else 0,
            'Failure_Rate': np.mean(metrics_list['Failure_Rate']) if metrics_list['Failure_Rate'] else 0,
        }

    output_file_path = os.path.join(output_dir, "metrics_uav123.txt")
    with open(output_file_path, 'w') as f:
        f.write("Overall Average Metrics:\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\nAttribute-based Average Metrics:\n")
        for attr, metrics in averaged_attribute_metrics.items():
            f.write(f"Attribute: {attr}\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")

        # f.write("\nMetrics per Sequence:\n")
        # for seq, metrics in all_sequence_metrics.items():
        #     f.write(f"Sequence: {seq}\n")
        #     for metric, value in metrics.items():
        #         f.write(f"  {metric}: {value:.4f}\n")

    # # Plotting
    # FIXED_ATTRIBUTE_ORDER = [
    #     'IV', 'SV', 'POC', 'FOC', 'OV', 'FM',
    #     'CM', 'BC', 'SOB', 'ARC', 'VC', 'LR'
    # ]
    # color_map = {attr: plt.cm.tab20(i) for i, attr in enumerate(FIXED_ATTRIBUTE_ORDER)}
    # color_map['Общий'] = 'black'  # Специальный цвет для общих метрик
    # color_map['Overall'] = 'black'

    if all_sequence_results:
        # Создаем большую фигуру 3x2
        fig = plt.figure(figsize=(28, 22))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

        # Создаем единую цветовую схему для всех атрибутов
        all_attributes = list(averaged_attribute_metrics.keys())
        color_map = {attr: plt.cm.tab20(i / len(all_attributes)) for i, attr in enumerate(sorted(all_attributes))}
        color_map['Общий'] = 'black'  # Специальный цвет для общих метрик

        # 1. График успешности (Success Plot) - левый верхний
        ax1 = fig.add_subplot(gs[0, 0])
        first_seq = next(iter(all_sequence_results))
        iou_thresholds = all_sequence_results[first_seq]['IoU_Thresholds']
        num_thresholds = len(iou_thresholds)

        # Общая кривая
        overall_arrays = []
        for res in all_sequence_results.values():
            arr = res.get('Success_Rates_at_Thresholds', [])
            if len(arr) == num_thresholds:
                overall_arrays.append(arr)
        overall_success_rates = np.mean(overall_arrays, axis=0) if overall_arrays else np.zeros(num_thresholds)
        ax1.plot(iou_thresholds, overall_success_rates, color=color_map['Общий'],
                 linewidth=3, linestyle='-', label='Общий')

        # Кривые для атрибутов
        valid_attrs = []
        attr_success_curves = {}
        for attr in all_attributes:
            attr_arrays = []
            for seq_name, attrs in uav123_attributes.items():
                if attrs.get(attr) == 1 and seq_name in all_sequence_results:
                    arr = all_sequence_results[seq_name].get('Success_Rates_at_Thresholds', [])
                    if len(arr) == num_thresholds:
                        attr_arrays.append(arr)

            if not attr_arrays:
                continue

            attr_success_rates = np.mean(attr_arrays, axis=0)
            ax1.plot(iou_thresholds, attr_success_rates, color=color_map[attr],
                     linewidth=2, linestyle='--', label=attr)
            valid_attrs.append(attr)
            attr_success_curves[attr] = attr_success_rates

        ax1.set_xlabel('Порог IoU', fontsize=12)
        ax1.set_ylabel('Успешность (%)', fontsize=12)
        ax1.set_title('График успешности по атрибутам (IoU)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', fontsize=10)

        # 2. График точности (Precision Plot) - правый верхний
        ax2 = fig.add_subplot(gs[0, 1])
        precision_thresholds = all_sequence_results[first_seq]['Precision_Thresholds']
        num_prec_thresholds = len(precision_thresholds)

        # Общая кривая
        overall_arrays = []
        for res in all_sequence_results.values():
            arr = res.get('Precision_at_Thresholds', [])
            if len(arr) == num_prec_thresholds:
                overall_arrays.append(arr)
        overall_precision_rates = np.mean(overall_arrays, axis=0) if overall_arrays else np.zeros(num_prec_thresholds)
        ax2.plot(precision_thresholds, overall_precision_rates, color=color_map['Общий'],
                 linewidth=3, linestyle='-', label='Общий')

        # Кривые для атрибутов
        for attr in valid_attrs:
            attr_arrays = []
            for seq_name, attrs in uav123_attributes.items():
                if attrs.get(attr) == 1 and seq_name in all_sequence_results:
                    arr = all_sequence_results[seq_name].get('Precision_at_Thresholds', [])
                    if len(arr) == num_prec_thresholds:
                        attr_arrays.append(arr)

            if not attr_arrays:
                continue

            attr_precision_rates = np.mean(attr_arrays, axis=0)
            ax2.plot(precision_thresholds, attr_precision_rates, color=color_map[attr],
                     linewidth=2, linestyle='--', label=attr)

        ax2.set_xlabel('Порог ошибки центра (пиксели)', fontsize=12)
        ax2.set_ylabel('Точность (%)', fontsize=12)
        ax2.set_title('График точности по атрибутам (ошибка центра)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)

        # 3. Графики метрик (средние значения) - левый нижний
        ax3 = fig.add_subplot(gs[1, 0])

        # Вычисляем Success Rate@50
        idx_50 = np.argmin(np.abs(iou_thresholds - 0.5))
        success_50 = {'Общий': overall_success_rates[idx_50]}
        for attr in valid_attrs:
            success_50[attr] = attr_success_curves[attr][idx_50]

        # Подготовка данных для графиков
        metrics_data = {
            'Avg_IoU': {
                'Общий': overall_metrics['Avg_IoU'],
                **{attr: averaged_attribute_metrics[attr]['Avg_IoU'] for attr in valid_attrs}
            },
            'Avg_P': {
                'Общий': overall_metrics['Avg_P'],
                **{attr: averaged_attribute_metrics[attr]['Avg_P'] for attr in valid_attrs}
            }
        }

        # Создаем сетку 2x1 внутри этого квадранта
        gs_left = gs[1, 0].subgridspec(2, 1, hspace=0.3)
        ax3_1 = fig.add_subplot(gs_left[0])
        ax3_2 = fig.add_subplot(gs_left[1])

        # Plotting
        if all_sequence_results:
            # Создаем большую фигуру с сеткой 5x2 (5 строк, 2 столбца)
            fig = plt.figure(figsize=(30, 40))
            gs = fig.add_gridspec(5, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1, 1.5],
                                  hspace=0.5, wspace=0.3)

            # Создаем единую цветовую схему для всех атрибутов
            all_attributes = list(averaged_attribute_metrics.keys())
            color_map = {attr: plt.cm.tab20(i) for i, attr in enumerate(sorted(all_attributes))}
            color_map['Overall'] = 'black'  # Специальный цвет для общих метрик

            # 1. Success Plot (IoU) - (0,0)
            ax1 = fig.add_subplot(gs[0, 0])
            first_seq = next(iter(all_sequence_results))
            iou_thresholds = all_sequence_results[first_seq]['IoU_Thresholds']

            # Общая кривая
            overall_success = np.mean([res['Success_Rates_at_Thresholds']
                                       for res in all_sequence_results.values()], axis=0)
            ax1.plot(iou_thresholds, overall_success, color=color_map['Overall'],
                     linewidth=3, linestyle='-', label='Overall')

            # Кривые для атрибутов
            valid_attrs = []
            attr_success_curves = {}
            for attr in all_attributes:
                seqs_with_attr = [seq for seq, attrs in uav123_attributes.items()
                                  if attrs.get(attr) == 1 and seq in all_sequence_results]

                if not seqs_with_attr:
                    continue

                attr_arrays = [all_sequence_results[seq]['Success_Rates_at_Thresholds']
                               for seq in seqs_with_attr]
                attr_success = np.mean(attr_arrays, axis=0)

                ax1.plot(iou_thresholds, attr_success, color=color_map[attr],
                         linewidth=2, linestyle='--', label=attr)
                valid_attrs.append(attr)
                attr_success_curves[attr] = attr_success

            ax1.set_xlabel('IoU Threshold', fontsize=16)
            ax1.set_ylabel('Success Rate', fontsize=16)
            ax1.set_title('Success Plot (IoU) by Attribute', fontsize=18)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='lower left', fontsize=14)

            # 2. Precision Plot - (0,1)
            ax2 = fig.add_subplot(gs[0, 1])
            precision_thresholds = all_sequence_results[first_seq]['Precision_Thresholds']

            # Общая кривая
            overall_precision = np.mean([res['Precision_at_Thresholds']
                                         for res in all_sequence_results.values()], axis=0)
            ax2.plot(precision_thresholds, overall_precision, color=color_map['Overall'],
                     linewidth=3, linestyle='-', label='Overall')

            # Кривые для атрибутов
            for attr in valid_attrs:
                seqs_with_attr = [seq for seq, attrs in uav123_attributes.items()
                                  if attrs.get(attr) == 1 and seq in all_sequence_results]

                attr_arrays = [all_sequence_results[seq]['Precision_at_Thresholds']
                               for seq in seqs_with_attr]
                attr_precision = np.mean(attr_arrays, axis=0)

                ax2.plot(precision_thresholds, attr_precision, color=color_map[attr],
                         linewidth=2, linestyle='--', label=attr)

            ax2.set_xlabel('Center Error Threshold (pixels)', fontsize=16)
            ax2.set_ylabel('Precision', fontsize=16)
            ax2.set_title('Precision Plot by Attribute', fontsize=18)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='lower right', fontsize=14)

            # 3. Bar Plots - средний ряд
            # Вычисляем Success Rate@50
            idx_50 = np.argmin(np.abs(iou_thresholds - 0.5))
            success_50 = {'Overall': overall_success[idx_50]}
            for attr in valid_attrs:
                success_50[attr] = attr_success_curves[attr][idx_50]

            # Подготовка данных для графиков
            metrics_to_plot = [
                ('Avg_IoU', 'Average IoU'),
                ('Avg_P', 'Average Precision'),
                ('Failure_Rate', 'Failure Rate'),
                ('Success@50', 'Success Rate@50'),
                ('AUC_IoU', 'AUC for Success Plot'),
                ('AUC_Precision', 'AUC for Precision Plot')
            ]

            # Создаем 6 субграфиков для метрик
            axes = [
                fig.add_subplot(gs[1, 0]),  # (1,0) Avg_IoU
                fig.add_subplot(gs[1, 1]),  # (1,1) Avg_P
                fig.add_subplot(gs[2, 0]),  # (2,0) Failure_Rate
                fig.add_subplot(gs[2, 1]),  # (2,1) Success@50
                fig.add_subplot(gs[3, 0]),  # (3,0) AUC_IoU
                fig.add_subplot(gs[3, 1])  # (3,1) AUC_Precision
            ]

            for ax, (metric_key, title) in zip(axes, metrics_to_plot):
                # Собираем данные
                if metric_key == 'Success@50':
                    data = success_50
                    overall_value = success_50['Overall']
                else:
                    data = {'Overall': overall_metrics[metric_key]}
                    data.update({attr: averaged_attribute_metrics[attr][metric_key] for attr in valid_attrs})
                    overall_value = overall_metrics[metric_key]

                # Удаляем Overall из данных для столбцов
                attrs = [a for a in data.keys() if a != 'Overall']
                values = [data[a] for a in attrs]

                # Сортируем по значению метрики
                if metric_key == 'Failure_Rate':
                    sorted_indices = np.argsort(values)  # Для Failure Rate лучше меньше
                else:
                    sorted_indices = np.argsort(values)[::-1]  # Для остальных лучше больше

                sorted_attrs = [attrs[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]

                # Создаем столбцы
                bar_colors = [color_map[attr] for attr in sorted_attrs]
                bars = ax.bar(sorted_attrs, sorted_values, color=bar_colors)

                # Настройки графика
                ax.set_title(title, fontsize=18)
                ax.set_ylabel(metric_key, fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=14)
                ax.tick_params(axis='y', labelsize=12)

                # Добавляем значения на столбцы с увеличенным отступом
                for bar in bars:
                    height = bar.get_height()
                    offset = 0.03 * (max(values) - min(values)) if values else 0.01
                    va = 'bottom' if height > 0 else 'top'
                    y_pos = height + offset if va == 'bottom' else height - offset
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            f'{height:.3f}',
                            ha='center', va=va, fontsize=14)

                # Добавляем линию общего среднего (красная пунктирная)
                ax.axhline(y=overall_value, color='red', linestyle='--',
                           linewidth=2, alpha=0.9)

                # Подпись для линии
                ax.text(len(sorted_attrs) - 0.5, overall_value,
                        f'Overall: {overall_value:.3f}',
                        color='red', fontsize=14, ha='right', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 4. Описание атрибутов и метрик - два отдельных окна в последней строке
            # Описание атрибутов - (4,0)
            ax_attr_desc = fig.add_subplot(gs[4, 0])
            ax_attr_desc.set_axis_off()

            attribute_descriptions = [
                "IV: Illumination Variation - Significant illumination changes",
                "SV: Scale Variation - Bounding box ratio outside [0.5, 2] range",
                "POC: Partial Occlusion - Target partially occluded",
                "FOC: Full Occlusion - Target fully occluded",
                "OV: Out-of-View - Target partially leaves view",
                "FM: Fast Motion - Bounding box motion >20px between frames",
                "CM: Camera Motion - Abrupt camera movement",
                "BC: Background Clutter - Background similar to target",
                "SOB: Similar Object - Objects of similar shape/type nearby",
                "ARC: Aspect Ratio Change - Aspect ratio outside [0.5, 2] range",
                "VC: Viewpoint Change - Significant viewpoint change",
                "LR: Low Resolution - Bounding box <400 pixels"
            ]

            ax_attr_desc.text(0.05, 0.95, "UAV123 ATTRIBUTES DESCRIPTION",
                              fontsize=22, weight='bold', verticalalignment='top')

            for i, desc in enumerate(attribute_descriptions):
                ax_attr_desc.text(0.05, 0.85 - i * 0.065, desc,
                                  fontsize=18, verticalalignment='top')

            # Описание метрик - (4,1)
            ax_metrics_desc = fig.add_subplot(gs[4, 1])
            ax_metrics_desc.set_axis_off()

            metric_descriptions = [
                "Avg_IoU: Average Intersection over Union",
                "Avg_P: Average Precision (center error)",
                "Avg_NP: Average Normalized Precision",
                "Failure_Rate: Ratio of tracking failures",
                "Success@50: Success rate at IoU threshold 0.5",
                "AUC_IoU: Area Under Curve for Success Plot",
                "AUC_Precision: Area Under Curve for Precision Plot",
                "FPS: Frames Per Second (tracking speed)",
                "Center Error: Distance between predicted and true center"
            ]

            ax_metrics_desc.text(0.05, 0.95, "METRICS DESCRIPTION",
                                 fontsize=22, weight='bold', verticalalignment='top')

            for i, desc in enumerate(metric_descriptions):
                ax_metrics_desc.text(0.05, 0.85 - i * 0.065, desc,
                                     fontsize=18, verticalalignment='top')

            # Общий заголовок с ключевыми метриками
            fig.suptitle(
                f'UAV123 Benchmark Results - Comprehensive Analysis\n'
                f'Overall Metrics: '
                f'Avg_IoU = {overall_metrics["Avg_IoU"]:.3f} | '
                f'Avg_P = {overall_metrics["Avg_P"]:.3f} | '
                f'Avg_NP = {overall_metrics["Avg_NP"]:.3f} | '
                f'Failure_Rate = {overall_metrics["Failure_Rate"]:.3f} | '
                f'AUC_IoU = {overall_metrics["AUC_IoU"]:.3f}',
                fontsize=24, y=0.98
            )

            # Регулируем отступы и сохраняем
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            output_file_path = os.path.join(output_dir, "uav123_benchmark_results.png")
            plt.savefig(output_file_path, dpi=150, bbox_inches='tight')
            plt.close()

    return overall_metrics, averaged_attribute_metrics, all_sequence_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to tracker config file")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to tracker checkpoint file")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root directory of the UAV123 dataset (e.g., .../UAV123_10fps/data_seq/UAV123_10fps)")
    parser.add_argument('--anno_root', type=str, required=True,
                        help="Root directory of the UAV123 annotations (e.g., .../UAV123_10fps/anno/UAV123_10fps)")
    parser.add_argument('--output_dir', type=str, default="results_uav123", help="Directory to save results")
    parser.add_argument('--matlab_config', type=str, required=True, help="Path to configSeqs.m file")
    parser.add_argument('--annotation_txt', type=str, default="",
                        help="Path to DatasetAnnotation.txt file (deprecated, use individual attribute files)")
    parser.add_argument('--visualize', action='store_true', help="Enable visualization during tracking")
    parser.add_argument('--debug', action='store_true', help="Enable debug prints for dataset structure")
    args = parser.parse_args()

    overall_metrics, averaged_attribute_metrics, all_sequence_metrics = run_tracking_evaluation(
        args.config,
        args.ckpt,
        args.data_root,
        args.anno_root,
        args.output_dir,
        args.matlab_config,
        args.annotation_txt,  # This argument is now effectively ignored for attribute parsing
        args.visualize,
        args.debug
    )
    print("Tracking complete. Metrics saved to", os.path.join(args.output_dir, "metrics_uav123.txt"))
    print("Plots saved to", args.output_dir)

'''
IV   Illumination Variation: the illumination of the target changes significantly.
SV   Scale Variation: the ratio of initial and at least one subsequent bounding box is outside the range [0.5, 2].
POC  Partial Occlusion: the target is partially occluded.
FOC  Full Occlusion: the target is fully occluded.
OV   Out-of-View: some portion of the target leaves the view.
FM   Fast Motion: motion of the ground-truth bounding box is larger than 20 pixels between two consecutive frames.
CM   Camera Motion: abrupt motion of the camera.
BC   Background Clutter: the background near the target has similar appearance as the target.
SOB  Similar Object: there are objects of similar shape or same type near the target.
ARC  Aspect Ratio Change: the fraction of ground-truth aspect ratio in the first frame and at least one subsequent frame is outside the range [0.5, 2].
VC   Viewpoint Change: viewpoint affects target appearance significantly.
LR   Low Resolution: at least one ground-truth bounding box has less than 400 pixels.
'''
