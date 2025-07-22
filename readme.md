conda create -n XXX python=3.10
conda activate XXX
cd XXX
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
pip install -r requirements_demo.txt
pip install -e .
cd XXX
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install -e .

download:
skyseg.onnx:https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx
path:/segmentProject/skyseg.onnx
sam_vit_h_4b8939.pth:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
path:/segmentProject/models/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth
(vggt)model.pt:https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
path:/.cache/torch/hub/checkpoints/model.pt