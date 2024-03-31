const { app, BrowserWindow , ipcMain} = require('electron/main')
const path = require('node:path')
const { dialog } = require('electron')
const { PythonShell } = require('python-shell')

async function handleVideoFileOpen() {
  const { canceled, filePaths } = await dialog.showOpenDialog()
  if (!canceled) {
    return filePaths[0]
  }
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
  createWindow()
  app.on('activate', ()=>{
    if (BrowserWindow.getAllWindows().length === 0){
        createWindow()
    }
  })
})

const options = {
    mode: 'text',
    //pythonPath: __dirname,
    pythonOptions: ['-u'],
    scriptPath: __dirname + '/python',
    args:[]
}

ipcMain.handle('python_detect', async (event, data)=>{
    //console.log('data: ' + str(data))
    fname = data.filename
    options.args = [fname]
    PythonShell.run('detect.py', options)
        .then((response)=>{
        event.sender.send("return_data", response);
        })
        .catch((error) =>{
            console.log(error);
        })
})

ipcMain.handle('python_analyze', async (event, data)=>{
    PythonShell.run('analyze.py', options)
        .then((response)=>{
        event.sender.send("return_data", response);
        })
        .catch((error) =>{
            console.log(error);
        })
})
