python test/test_img_attr.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --n_epochs=50 \
       --n_categories=1601 \
       --lr=1e-4 \
       --exp_name=image_attribute \
       --img_attr_pretrained=YOUR_PROJECT_PATH/logs/image_attribute_0108_153147/snapshots/49.pkl \
       --pretrained=YOUR_PROJECT_PATH/logs/image_attribute_0108_153147/snapshots/49.pkl
