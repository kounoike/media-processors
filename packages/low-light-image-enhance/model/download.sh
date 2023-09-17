#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/370_Semantic-Guided-Low-Light-Image-Enhancement/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz semantic_guided_llie_HxW.onnx
rm resources.tar.gz

echo Download finished.
