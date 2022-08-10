# Adversarial Machine Learning

## Dataset

### CIFAR-10/CIFAR100
``` python
# CIFAR-10
mean = [0.491, 0.482, 0.447]
std  = [0.247, 0.243, 0.262]
# CIFAR-100
mean = [0.507, 0.487, 0.441]
std  = [0.267, 0.256, 0.276]
```
### CINIC-10
``` bash
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
```

```python

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

data_path = '/path/to/cinic'
mean = [0.47889522, 0.47227842, 0.43047404]
std = [0.24205776, 0.23828046, 0.25874835]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=cinic_mean,std=cinic_std)
])
dataset = ImageFolder(os.path.join(data_path, "train"), transform=transform)
```

### Tiny-ImageNet-200

``` bash
#!/bin/bash
# https://gist.githubusercontent.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4/raw/5a9ed1c597f5a066bcd89464092730e025f00ed7/tinyimagenet.sh
# download and unzip dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"

# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
echo "done"
```

