# WSUnet
snippets of weakly supervised U-net, trainable with global labels and LSE pooling function

# How to use
1. Build a decoder for your encoder. For an example look at [SaliencyUNet](https://github.com/maxunterwegs/WSUnet/blob/22f5ca8e0a2ba0cbf356d79ba9003eb06301ef68/saliency_map_net.py#L666) class in saliency_map_net.py.
Note how in the forward function we use a custom pooling function, to reduce the saliency maps to a class score.

2. This pooling function is the [LSE_LBA](model.py) pooling function, implemented in model.py as CustomPooling class.

3. classification_training.py is an example on how to train this architecture. 
Note that there are many options for training. You can load a pretrained encoder network and freeze the weights. 

#How to adapt to 3D
1. replace 2D operations with 3D operations
2. modify the LSE_LBA pooling function to 3D. 
3. reduce capacity to fit on GPU

# Questions
Contact me at maximilian.moeller@tum.de
