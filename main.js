const { app, BrowserWindow , ipcMain} = require('electron/main')
const path = require('node:path')
const { dialog } = require('electron')
const { PythonShell } = require('python-shell')

async function handleVideoFileOpen() {
  const { canceled, filePaths } = await dialog.showOpenDialog(null, {
    properties: ['openFile'],
    defaultPath: 'yolo',
    title: 'ビデオファイルを選択',
    filters: [
      {name:'video file', extensions:['mp4']}
    ]
  })
  if (!canceled) {
    return filePaths[0]
  }
}

const options = {
  mode: 'text',
  //pythonPath: __dirname,
  pythonOptions: ['-u'],
  scriptPath: __dirname + '/python',
  args:[]
}

async function birdDetect(event, data) {
  fname = data.filename
  options.args = [fname]
  PythonShell.run('detect.py', options)
      .then((response)=>{
      event.sender.send("return_data", response);
      })
      .catch((error) =>{
          console.log(error);
      })
}

async function birdAnalyze(event, data) {
  PythonShell.run('analyze.py', options)
      .then((response)=>{
      event.sender.send("return_data", response);
      })
      .catch((error) =>{
          console.log(error);
      })
}

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
        preload: path.join(__dirname, 'preload.js')
    }
  })

  win.loadFile('index.html')
  // win.webContents.openDevTools()
}

app.whenReady().then(() => {
  ipcMain.handle('dialog:openFile', handleVideoFileOpen)
  ipcMain.handle('python:detect', birdDetect)
  ipcMain.handle('python:analyze', birdAnalyze)
  createWindow()
  app.on('activate', ()=>{
    if (BrowserWindow.getAllWindows().length === 0){
        createWindow()
    }
  })
})
