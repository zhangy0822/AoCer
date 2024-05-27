txt_img_matching_pretrained_ckpt=YOUR_PROJECT_PATH/logs/text_image_matching_0108_220651/snapshots/49.pkl
pretrained_ckpt=YOUR_PROJECT_PATH/logs/text_image_matching_0108_220651/snapshots/49.pkl

python test/test_txt_img_matching.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --max_turns=4 \
       --n_epochs=50 \
       --lr=5e-4 \
       --exp_name=text_image_matching_0825_184459 \
       --txt_img_matching_pretrained=$txt_img_matching_pretrained_ckpt \
       --pretrained=$pretrained_ckpt
