# CIFAR10-Tensorflow-single-image-test
Training by cifar10 image files(.jpg);
Used to test single imgae file(step3-evaluation.py) or a batch of image files(step2-Accuracy.py);

The test accuracy and training loss chart:

![Test_Accuracy](https://github.com/KimMeen/CIFAR10-Tensorflow-Single-Image-Test/raw/master/Test_accuracy.png)

![Training_Loss](https://github.com/KimMeen/CIFAR10-Tensorflow-Single-Image-Test/raw/master/training_loss.png)

important files list:
1.genFileList.py - used to create 'train.txt' and 'test.txt'.
2.icifar10.py - used to import training and testing data,include some data enhancement functions.
3.step1_classification.py - structure of network and used to train the model.
4.step2_Accuarcy.py - used all of the test data to evaluate the accuarcy of network.
3.step3_evaluation.py - used model to test single image file,which in the test_images folder.


