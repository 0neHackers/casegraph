#!/usr/bin/env bash
# Local TigerGraph Community Edition for development (Savanna is the target for the demo).
# Normal machine:   docker run -d --name tigergraph --ulimit nofile=1000000:1000000 -p 14240:14240 -p 9000:9000 tigergraph/community:4.2.5
# Sandboxes that cap file descriptors below 65,535 (GSE refuses to start) can use the getrlimit shim below.
set -euo pipefail
docker run -d --name tigergraph -p 14240:14240 -p 9000:9000 tigergraph/community:4.2.5
gcc -shared -fPIC -O2 -o /tmp/fdshim.so "$(dirname "$0")/fdshim.c" -ldl
docker cp /tmp/fdshim.so tigergraph:/home/tigergraph/fdshim.so
docker exec -u root tigergraph chmod 755 /home/tigergraph/fdshim.so
docker exec -u tigergraph tigergraph bash -lc '
  export PATH=$PATH:/home/tigergraph/tigergraph/app/cmd
  for s in GSE GPE RESTPP; do
    gadmin config set $s.BasicConfig.Env "LD_PRELOAD=/home/tigergraph/fdshim.so:\$LD_PRELOAD; LD_LIBRARY_PATH=\$LD_LIBRARY_PATH; CPUPROFILE=/tmp/tg_cpu_profiler; CPUPROFILESIGNAL=34; MALLOC_CONF=prof:true,prof_active:false"
  done
  gadmin config apply -y && gadmin restart all -y'
