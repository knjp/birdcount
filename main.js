const { app, BrowserWindow , ipcMain} = require('electron/main')
const path = require('node:path')
const fs = require('fs')
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

async function handleCSVFileSave() {
  const  filePath = dialog.showSaveDialogSync(null,{
    title: 'ファイル保存',
    defaultPath: 'bird.csv',
    properties: ['showOverwriteConfirmation']
  })
  if (filePath === undefined) {
    return ({status: undefined})
  }
  try {
    const data = fs.readFileSync('yolo/birdstatus.csv')
    fs.writeFileSync(filePath, data)
    return ({
      status: true,
      path: filePath
    })
  }
  catch (error) {
    console.log(error.message)
    return ({status: false})
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
  saveVideo = data.saveVideo
  if(saveVideo === true)
    options.args = [fname, '--save_video']
  else
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
  options.args = []
  PythonShell.run('analyze.py', options)
      .then((response)=>{
      event.sender.send("return_data", response);
      })
      .catch((error) =>{
          console.log(error);
      })
}

async function birdAnaplotly(event, data) {
  options.args = []
  PythonShell.run('anaplotly.py', options)
      .then((response)=>{
      event.sender.send("return_data", response);
      })
      .catch((error) =>{
          console.log(error);
      })
}

async function subWindowOpen(event, data) {
  const subWindow = new BrowserWindow({
    title: 'Results of detection',
    width: 1200,
    height: 1000,
    webPreferences: {
      preload: path.join(__dirname, 'subwin/preload.js')
    }
  })
  subWindow.loadFile('subwin/index.html')
}

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1200,
    height: 1000,
    webPreferences: {
        preload: path.join(__dirname, 'preload.js')
    }
  })

  win.loadFile('index.html')
  // win.webContents.openDevTools()
}

app.whenReady().then(() => {
  ipcMain.handle('dialog:openVideoFile', handleVideoFileOpen)
  ipcMain.handle('dialog:saveCSVFile', handleCSVFileSave)
  ipcMain.handle('python:detect', birdDetect)
  ipcMain.handle('python:analyze', birdAnalyze)
  ipcMain.handle('python:scatter3d', birdAnaplotly)
  ipcMain.handle('window:analyzedVideo', subWindowOpen)
  createWindow()
  app.on('activate', ()=>{
    if (BrowserWindow.getAllWindows().length === 0){
        createWindow()
    }
  })
})

app.on('activate', () => {
  setInterval('printTime()', 1000)
})

