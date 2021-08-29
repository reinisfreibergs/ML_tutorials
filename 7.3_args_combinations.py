import subprocess
run_order = ["python 6.3_conv_net_task_ResNet.py -epochs 31",
             "python 6.3_conv_net_task_ResNet.py -epochs 31 -batch_size 8",
             "python 7.1_densenet_template.py -epochs 31",
             "python 7.2_shufflenet_unfinished.py -epochs 31",
             "python 6.3_conv_net_task_ResNet.py -epochs 101 -batch_size 8"]

for run in run_order:
    subprocess.run(run, shell = False)


