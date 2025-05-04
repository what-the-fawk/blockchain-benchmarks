#! /bin/bash

# run optimization process (from built image fabric-bo)
docker run -it --rm \
  -p 7050:7050 \
  -p 7051:7051 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  --privileged \
  
