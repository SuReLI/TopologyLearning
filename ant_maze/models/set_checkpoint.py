
import os

# Remove checkpoint file if it exist
if os.path.exists("checkpoint"):
    os.remove("checkpoint")

# Re-create the file and write into it
"""
model_checkpoint_path: "/home/disc/h.bonnavaud/Bureau/HAC_original/ant_environments/ant_reacher_3_levels/models/HAC.ckpt-99"
all_model_checkpoint_paths: "/home/disc/h.bonnavaud/Bureau/HAC_original/ant_environments/ant_reacher_3_levels/models/HAC.ckpt-99"
"""
current_directory_path = __file__[:-len("set_checkpoint.py")]

model_checkpoint_path = current_directory_path + "HAC.ckpt-99"
with open('checkpoint', 'w') as f:
    f.write('model_checkpoint_path: ' + model_checkpoint_path + "\n")
    f.write('all_model_checkpoint_paths: ' + model_checkpoint_path)
