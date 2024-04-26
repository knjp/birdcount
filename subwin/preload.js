const { contextBridge, ipcRenderer} = require('electron/renderer')

contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron
})

contextBridge.exposeInMainWorld('runpython', {
  scatter3d: async (data) => await ipcRenderer.invoke('python:scatter3d', data),
  on: (channel, func) => {
    ipcRenderer.once(channel, (event,data)=>func(data))
  }
})

contextBridge.exposeInMainWorld('electronAPI', {
  selectLoadVideoFile: () => ipcRenderer.invoke('dialog:openVideoFile'),
  saveCSVFile: () => ipcRenderer.invoke('dialog:saveCSVFile'),
  openVideoWindow: () => ipcRenderer.invoke('window:analyzedVideo')
})