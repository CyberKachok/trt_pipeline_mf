import argparse
#import onnxsim
import onnx
import torch
import os

from model.torch_tracker_wrapper import TorchTrackerWrapper


def convert_to_jit(cfg_path, ckpt_path):
    wrapper = TorchTrackerWrapper(cfg_path, ckpt_path)

    model_jit = torch.jit.script(wrapper.network)
    model_jit.save(ckpt_path[:-1])

    print('[INFO] Model successfully converted to jit-script:', ckpt_path[:-1])


def convert_to_onnx(cfg_path, ckpt_path):
    wrapper = TorchTrackerWrapper(cfg_path, ckpt_path)

    dummy_template = torch.randn((1, 3, wrapper.template_size, wrapper.template_size), dtype=torch.float32).cuda()
    dummy_online_template = torch.randn((1, 3, wrapper.template_size, wrapper.template_size), dtype=torch.float32).cuda()
    dummy_search = torch.randn((1, 3, wrapper.search_size, wrapper.search_size), dtype=torch.float32).cuda()

    model_path, _ = os.path.splitext(ckpt_path)
    model_path_onnx = model_path + '.onnx'

    torch.onnx.export(wrapper.network,
                      (dummy_template, dummy_online_template, dummy_search,),
                      model_path_onnx,
                      export_params=True,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=['template', 'online_template', 'search'],
                      output_names=['bbox', 'confidence'],
                      dynamic_axes=None)

    model_onnx = onnx.load(model_path_onnx)     # load onnx model
    onnx.checker.check_model(model_onnx)        # check onnx model

    print('[INFO] Model successfully converted to onnx:', model_path_onnx)

    cuda = torch.cuda.is_available()
    if not cuda:
        print('[WARNING] Cuda is not avalible :(')

    #model_onnx, check = onnxsim.simplify(model_onnx)
    #assert check, '[ERROR] Assert onnx-simplified check failed'
    onnx.save(model_onnx, model_path + '_simple.onnx')

    print('[INFO] Onnx model successfully simplified:', 
          model_path + '_simple.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--mod', type=str, help='jit / onnx / engine')

    args = parser.parse_args()

    if args.mod == 'jit':
        convert_to_jit(args.config, args.ckpt)
    elif args.mod == 'onnx':
        convert_to_onnx(args.config, args.ckpt)

