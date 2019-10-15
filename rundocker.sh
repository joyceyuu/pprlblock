# build docker image
docker build -t pprlblock .


# build docker container that limits RAM
docker run --name pprl -t -m 8g pprlblock


# remove this container once it is done
docker container rm --force pprl


# copy output file to host system
docker cp pprl:/result_n=461167.csv .

# copy file from remote fractal machine to local
# scp wan220@fractal-ev:psig/result_n=461167.csv ./    
