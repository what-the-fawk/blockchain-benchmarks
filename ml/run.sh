#! /bin/bash

# install dependencies
pip install -r requirements.txt
git clone https://github.com/hyperledger/fabric-samples.git ../fabric-samples
git clone https://github.com/hyperledger-caliper/caliper-benchmarks.git ../caliper-benchmarks
cd ..
npm install --only=prod @hyperledger/caliper-cli
npx caliper bind --caliper-bind-sut fabric:2.4"

# run the benchmark
