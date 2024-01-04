const express = require('express');
const { exec } = require('child_process');
const path = require('path');  // Thêm đoạn này để sử dụng module path
const app = express();
const port = 3000;

//Kết nối tới Frontend
app.use(express.static('Frontend'));
app.use(express.json());

//Kết nối tới Backend
const pythonScriptPath = "/Users/daomanh/Desktop/Fake-News-Detection/Backend/target/test.py";


// Đặt môi trường PYTHONPATH kết nối tới thư viện python
process.env.PYTHONPATH = "/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python"; // Thay thế "/đường/dẫn/tới/python/lib" bằng đường dẫn thực tế
console.log('Môi trường thực thi Python:', process.env.PYTHONPATH);


app.post('/checkArticle', (req, res) => {
  const text = req.body.text;

  // Thực hiện kịch bản Python của bạn với văn bản được cung cấp
  exec(`python3 ${pythonScriptPath} "${text}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Lỗi: ${error.message}`);
      return res.status(500).send('Lỗi Nội Server');
    }

    console.log(`Kết quả từ kịch bản Python: ${stdout}`);
    const result = stdout.trim();
    res.send({ result });
  });
});

app.listen(port, () => {
  console.log(`Máy chủ đang chạy tại http://localhost:${port}`);
});

const cors = require('cors');
app.use(cors());