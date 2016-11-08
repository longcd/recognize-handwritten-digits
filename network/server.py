# -*- coding: UTF-8 -*-

import numpy as np
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from network2 import Network
from network2 import CrossEntropyCost


#服务器端配置
HOST = 'localhost'
PORT = 8889


class JSONHandler(BaseHTTPRequestHandler):
    """处理接收到的POST请求"""
    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len).decode("utf-8");
        payload = json.loads(content);

        # print(payload['image'])
        # print(len(payload['image']))

        # 如果是预测请求，返回预测值
        if payload.get('predict'):
            try:
                result = self.predict(payload['image'])
                response = {"type":"test", "result":str(result)}
            except:
                response_code = 500
        else:
            response_code = 400


        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode("utf-8"))
        return

    def predict(self, x):
        f = open('./network2.json', "r")
        data = json.load(f)
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]


        a = np.array(x).reshape(784, -1)
        for b,w in zip(net.biases, net.weights):
            a = 1.0 / (1.0 + np.exp(-(np.dot(w, a) + b)))

        return np.argmax(a)


if __name__ == '__main__':
    httpd = HTTPServer((HOST, PORT), JSONHandler)

    try:
        #启动服务器
        print('Serving HTTP on port %s' % PORT)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()