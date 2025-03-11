In this README file, we take the Resnet-20 as the Student net and the Resnet-32 as the Teacher net.

Step 1：Produce the Teacher net 's pth :

	python produceModelPth.py --model resnet32

Step 2:  Produce the Proximal Teacher net and the corresponding Student net :

	python produceUnionATKDModelPth.py --model resnet20 --teachermode resnet32
	python produceUnionATKDModelPth.py --model mobilenetv2 --teachermode vgg13

