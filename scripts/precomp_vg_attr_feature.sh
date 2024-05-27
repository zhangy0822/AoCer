python datasets/precomp_vg_attr_feature.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --max_turns=10 \
       --n_epochs=50 \
       --lr=5e-4 \
       --exp_name=text_image_matching \
       --txt_img_matching_pretrained=YOUR_PROJECT_PATH/logs/text_image_matching_0108_220651/snapshots/49.pkl \
       --pretrained=YOUR_PROJECT_PATH/logs/text_image_matching_0108_220651/snapshots/49.pkl 
