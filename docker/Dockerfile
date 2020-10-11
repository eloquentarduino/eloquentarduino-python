FROM  python:3.8-slim

EXPOSE 3000
CMD cd /home
RUN apt-get install -y curl && \
    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh && \
    arduino-cli board listall && \
    arduino-cli core update-index && \
    arduino-cli core install arduino:mbed &&
    arduino-cli core install grumpyoldpizza:stm32l4 && \
    pip install --no-cache scipy eloquentarduino notebook
CMD jupyter notebook --ip=0.0.0.0 --port=3000 --allow-root
ENTRYPOINT /bin/sh