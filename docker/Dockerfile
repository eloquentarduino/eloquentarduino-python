FROM  python:3.8-slim

EXPOSE 8888
WORKDIR /home
ADD arduino-cli.yaml /home/arduino-cli.yaml
CMD cd /home
RUN apt-get update && \
    apt-get install -y gcc-multilib && \
    apt-get install -y curl && \
    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=/usr/bin sh && \
    arduino-cli board listall && \
    arduino-cli core update-index && \
    arduino-cli core install arduino:mbed && \
    arduino-cli core install grumpyoldpizza:stm32l4 && \
    pip install --no-cache scipy eloquentarduino notebook
ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
ENTRYPOINT jupyter notebook

# run as
# docker run -i -t --rm --privileged -p 8889:8888 -v /dev/bus/usb:/dev/bus/usb -v $(pwd):/home eloquentarduino-notebook