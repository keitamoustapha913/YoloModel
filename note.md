                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]  
 1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]  
 2 -1 1 6640 ultralytics.nn.modules.block.C3k2 [32, 64, 1, False, 0.25]  
 3 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]  
 4 -1 1 26080 ultralytics.nn.modules.block.C3k2 [64, 128, 1, False, 0.25]  
 5 -1 1 147712 ultralytics.nn.modules.conv.Conv [128, 128, 3, 2]  
 6 -1 1 87040 ultralytics.nn.modules.block.C3k2 [128, 128, 1, True]  
 7 -1 1 295424 ultralytics.nn.modules.conv.Conv [128, 256, 3, 2]  
 8 -1 1 346112 ultralytics.nn.modules.block.C3k2 [256, 256, 1, True]  
 9 -1 1 249728 ultralytics.nn.modules.block.C2PSA [256, 256, 1]  
 10 -1 1 337926 ultralytics.nn.modules.head.Classify [256, 6]  
YOLO11n-cls summary: 86 layers, 1,538,790 parameters, 1,538,790 gradients, 3.3 GFLOPs

                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]  
 1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]  
 2 -1 1 6640 ultralytics.nn.modules.block.C3k2 [32, 64, 1, False, 0.25]  
 3 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]  
 4 -1 1 26080 ultralytics.nn.modules.block.C3k2 [64, 128, 1, False, 0.25]  
 5 -1 1 147712 ultralytics.nn.modules.conv.Conv [128, 128, 3, 2]  
 6 -1 1 87040 ultralytics.nn.modules.block.C3k2 [128, 128, 1, True]  
 7 -1 1 295424 ultralytics.nn.modules.conv.Conv [128, 256, 3, 2]  
 8 -1 1 346112 ultralytics.nn.modules.block.C3k2 [256, 256, 1, True]  
 9 -1 1 164608 ultralytics.nn.modules.block.SPPF [256, 256, 5]  
 10 -1 1 249728 ultralytics.nn.modules.block.C2PSA [256, 256, 1]  
 11 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']  
 12 [-1, 6] 1 0 ultralytics.nn.modules.conv.Concat [1]  
 13 -1 1 111296 ultralytics.nn.modules.block.C3k2 [384, 128, 1, False]  
 14 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']  
 15 [-1, 4] 1 0 ultralytics.nn.modules.conv.Concat [1]  
 16 -1 1 32096 ultralytics.nn.modules.block.C3k2 [256, 64, 1, False]  
 17 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]  
 18 [-1, 13] 1 0 ultralytics.nn.modules.conv.Concat [1]  
 19 -1 1 86720 ultralytics.nn.modules.block.C3k2 [192, 128, 1, False]  
 20 -1 1 147712 ultralytics.nn.modules.conv.Conv [128, 128, 3, 2]  
 21 [-1, 10] 1 0 ultralytics.nn.modules.conv.Concat [1]  
 22 -1 1 378880 ultralytics.nn.modules.block.C3k2 [384, 256, 1, True]  
[detect] self.reg_max 16
23 [16, 19, 22] 1 430867 ultralytics.nn.modules.head.Detect [1, [64, 128, 256]]  
YOLO11n summary: 181 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs

uv run python profile_backbone.py --device cuda --height 640 --width 640 --iterations 200 --warmup 100 --batch-size 1
Device: cuda
GPU: NVIDIA GeForce RTX 2050
Dtype: torch.float32
Input: (1, 3, 640, 640)
Output: (1, 256, 20, 20)
Parameters: 1,200,864
MACs: 3,740,518,400 (3.740518 GMACs)
GFLOPs: 7.481037 (2 FLOPs per MAC)

Per-stage MACs:
0: 176.947 MMACs -> (1, 16, 320, 320)
1: 471.859 MMACs -> (1, 32, 160, 160)
2: 163.840 MMACs -> (1, 64, 160, 160)
3: 943.718 MMACs -> (1, 64, 80, 80)
4: 163.840 MMACs -> (1, 128, 80, 80)
5: 943.718 MMACs -> (1, 128, 40, 40)
6: 137.626 MMACs -> (1, 128, 40, 40)
7: 471.859 MMACs -> (1, 256, 20, 20)
8: 137.626 MMACs -> (1, 256, 20, 20)
9: 129.485 MMACs -> (1, 256, 20, 20)

Speed:
Latency: 3.464 ms/image
FPS: 288.68 images/second
