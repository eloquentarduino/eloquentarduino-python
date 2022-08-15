import os
import json
import signal
import bottle
from time import sleep
from bottle import Bottle
from threading import Thread


class LabelStudioBottle(Bottle):
    """

    """
    def __init__(self, studio, dataset):
        super(LabelStudioBottle, self).__init__()

        self.studio = studio
        self.dataset = dataset

        self.get('/', callback=self.get_dataset)
        self.post('/', callback=self.update_dataset)
        self.route('/', callback=self.preflight, method=['OPTIONS'])
        self.route('/', callback=self.shutdown, method=['DELETE'])

    def preflight(self):
        """
        Add CORS headers
        """
        bottle.response.add_header('Access-Control-Allow-Origin', '*')
        bottle.response.add_header('Access-Control-Allow-Headers', '*')
        bottle.response.add_header('Access-Control-Allow-Methods', '*')
        bottle.response.add_header('Content-Type', 'application/json')

    def get_dataset(self):
        """

        """
        self.preflight()

        return json.dumps(self.dataset)

    def update_dataset(self):
        """

        """
        self.preflight()

        body = bottle.request.body.read().decode('utf-8')

        if len(body) > 0:
            data = json.loads(body)
            self.studio.update_dataset(data)

        return json.dumps('OK')

    def shutdown(self):
        """
        Stop server
        """
        self.preflight()

        def shutdown_server():
            sleep(2)
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)

        Thread(target=shutdown_server).start()

        return json.dumps('OK')


