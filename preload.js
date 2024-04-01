const { contextBridge, ipcRenderer} = require('electron/renderer')

contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron
})

contextBridge.exposeInMainWorld('runpython', {
  detect: async (data) => await ipcRenderer.invoke('python:detect', data),
  analyze: async (data) => await ipcRenderer.invoke('python:analyze', data),
  on: (channel, func) => {
    ipcRenderer.once(channel, (event,data)=>func(data))
  }
})

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('dialog:openFile')
})