# build docker image
docker build -t pprlblock .


# build docker container that limits RAM
docker run --name pprl -t -m 8g pprlblock


# remove this container once it is done
docker container rm --force pprl
