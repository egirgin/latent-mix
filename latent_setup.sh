cd latent_compostion

bash resources/download_resources.sh

pip install ninja

# additional requirements for finetuning
pip install lpips
# face landmarks model
mkdir -p resources/dlib
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat.bz2 resources/dlib
bunzip2 resources/dlib/shape_predictor_68_face_landmarks.dat.bz2
# identity loss model from pixel2style2pixel
gdown --id 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn # pretrained model
mkdir -p resources/psp
mv model_ir_se50.pth resources/psp