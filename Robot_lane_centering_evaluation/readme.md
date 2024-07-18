## Objective

Auto evaluate the robot centering error, and generate the trojectory map

## How to use

1. Record the 360-cam video, output the dewrap video
2. measure the target point using ImageJ software, type into the config file
3. run the commend

   ```bash
   python3 evaluation_lane_centering.py \
       --config Data/case2_hdbscan_test2/config.yaml \
   ```

   then, the trojectory map and the the error csv file will be saved.
   and save the missing / identification error frame number.
4. You can manual check the identification result. If some error case, use `evaluation_one_frame.py `code to manual annote the missing frame, type the frame index in the yaml file.

   ```bash
   python3 evaluation_one_frame.py
   ```

---
