#./bin/giexec --model=./models/googlenet/googlenet.caffemodel --deploy=./models/googlenet/googlenet.prototxt --output=prob --batch=256

network=vgg16
batch_size=512
device=1
giexec --model=./models/$network/$network.caffemodel --deploy=./models/$network/deploy.prototxt --output=prob --batch=$batch_size --device=$device
