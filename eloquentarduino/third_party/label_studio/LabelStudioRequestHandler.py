import json
from http.server import BaseHTTPRequestHandler


class LabelStudioRequestHandler(BaseHTTPRequestHandler):
    """
    Handle HTTP requests from Trainset studio
    """
    DATASET = None

    def do_OPTIONS(self):
        """
        Pre-flight request
        """
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        """

        """
        if self.path == '/dataset':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(self.DATASET), 'utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """

        """
        content_len = int(self.headers.getheader('content-length', 0))
        post_body = self.rfile.read(content_len)
        
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(bytes('OK', 'utf-8'))