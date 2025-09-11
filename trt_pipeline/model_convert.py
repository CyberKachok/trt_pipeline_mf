import argparse
import onnx
import torch
import os

from model.torch_tracker_wrapper import TorchTrackerWrapper


def convert_to_jit(cfg_path, ckpt_path):
    """Convert a PyTorch checkpoint to TorchScript.

    Returns the path to the saved TorchScript model (.pt).
    """
    wrapper = TorchTrackerWrapper(cfg_path, ckpt_path)
    wrapper.network.eval()
    model_jit = torch.jit.script(wrapper.network.cpu())

    model_path, _ = os.path.splitext(ckpt_path)
    jit_path = model_path + '.pt'
    model_jit.save(jit_path)
    print('[INFO] Model successfully converted to jit-script:', jit_path)
    return jit_path


def convert_to_onnx(cfg_path, ckpt_path):
    """Export the tracker network to ONNX and return the model path."""
    wrapper = TorchTrackerWrapper(cfg_path, ckpt_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper.network.to(device).eval()

    dummy_template = torch.randn((1, 3, wrapper.template_size, wrapper.template_size),
                                 dtype=torch.float32, device=device)
    dummy_online_template = torch.randn((1, 3, wrapper.template_size, wrapper.template_size),
                                        dtype=torch.float32, device=device)
    dummy_search = torch.randn((1, 3, wrapper.search_size, wrapper.search_size),
                               dtype=torch.float32, device=device)

    model_path, _ = os.path.splitext(ckpt_path)
    model_path_onnx = model_path + '.onnx'

    torch.onnx.export(wrapper.network,
                      (dummy_template, dummy_online_template, dummy_search),
                      model_path_onnx,
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=['template', 'online_template', 'search'],
                      output_names=['bbox', 'confidence'],
                      dynamic_axes=None)

    model_onnx = onnx.load(model_path_onnx)
    onnx.checker.check_model(model_onnx)
    print('[INFO] Model successfully converted to onnx:', model_path_onnx)

    if device.type != 'cuda':
        print('[WARNING] CUDA is not available; exported ONNX on CPU')

    return model_path_onnx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--mod', type=str, help='jit / onnx')

    args = parser.parse_args()

    if args.mod == 'jit':
        convert_to_jit(args.config, args.ckpt)
    elif args.mod == 'onnx':
        convert_to_onnx(args.config, args.ckpt)

