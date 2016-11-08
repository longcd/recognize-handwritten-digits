var digitRecognizer = {
    CANVAS_WIDTH: 280,
    TRANSLATED_WIDTH: 28,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
    BATCH_SIZE: 1,

    // 服务器端参数
    PORT: "8889",
    HOST: "http://localhost",

    // 颜色变量
    BLACK: "#000000",
    BLUE: "#0000ff",

    // 加载页面
    onLoadFunction() {
        this.resetCanvas();
    },

    // 重置
    resetCanvas() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        this.data = []
        ctx.fillStyle = this.BLACK; // 设置或返回用于填充绘画的颜色、渐变或模式。
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH); // 绘制“已填色”的矩形。默认的填充颜色是黑色。

        // 初始化data
        var matrixSize = 784;
        while(matrixSize > 0) {
            this.data.push(0);
            matrixSize--;
        }
        this.drawGrid(ctx);

        // 绑定事件操作
        canvas.onmousemove = function(e) { this.onMouseMove(e, ctx, canvas) }.bind(this); // 当鼠标指针移动到元素上时触发
        canvas.onmousedown = function(e) { this.onMouseDown(e, ctx, canvas) }.bind(this); // 当元素上按下鼠标按钮时触发。
        canvas.onmouseup = function(e) { this.onMouseUp(e, ctx) }.bind(this); // 当在元素上释放鼠标按钮时触发
    },

    // 初始化网格
    drawGrid(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE; // 设置笔触的颜色(蓝)

            // 竖线
            ctx.beginPath(); // 起始一条路径，或重置当前路径
            ctx.moveTo(x, 0); // 把路径移动到画布中的指定点(x,0)，不创建线条
            ctx.lineTo(x, this.CANVAS_WIDTH); // 添加一个新点，然后在画布中创建从该点到最后指定点(x,this.CANVAS_WIDTH)的线条
            ctx.stroke(); // 绘制已定义的路径

            // 横线
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    // 鼠标移动
    onMouseMove(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        // clientX 事件属性返回当事件被触发时鼠标指针向对于浏览器页面（或客户区）的水平坐标。
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp(e) {
        canvas.isDrawing = false;
    },

    // 填充方格
    fillSquare(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH); // Math.floor下取整
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        // 存储手写输入数据
        this.data[yPixel * this.TRANSLATED_WIDTH + xPixel] = 1;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },


    // 发送预测请求
    test() {
        if (this.data.indexOf(1) < 0) {
            alert("请写入一个数字以进行测试");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    // 处理服务器响应
    receiveResponse(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            alert("The neural network predicts you wrote a \'" + responseJSON.result + '\'');
        }
    },

    onError(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, false); // open() 初始化 HTTP 请求参数，例如 URL 和 HTTP 方法，但是并不发送请求。
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length); // setRequestHeader() 方法指定了一个 HTTP 请求的头部
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
}