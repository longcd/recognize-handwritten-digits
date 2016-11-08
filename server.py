# -*- coding: UTF-8 -*-

from http.server import BaseHTTPRequestHandler, HTTPServer
from logistic_regression.logistic_regression import LogisticRegression
import json
import pickle

#服务器端配置
HOST = 'localhost'
PORT = 8888

class JSONHandler(BaseHTTPRequestHandler):
    """处理接收到的POST请求"""
    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len).decode("utf-8");
        payload = json.loads(content);

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
        
        with open('./logistic_regression/logistic_regression.pkl', 'rb') as f:
            clf = pickle.load(f)

            pred_y = list()
            for number in range(0, 10):
                pred_y.append(clf[number].predict(np.reshape(x, (1, -1))))

            result = np.argmax(pred_y)

            return result



if __name__ == '__main__':
    httpd = HTTPServer((HOST, PORT), JSONHandler)

    try:
        #启动服务器
        print('Serving HTTP on port %s' % PORT)
        httpd.serve_forever() # # 设置一直监听并接收请求
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()