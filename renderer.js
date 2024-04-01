const information = document.getElementById('info')
information.innerText = `This app is using Chrome (v${versions.chrome()}), Node.js (v${versions.node()}), and Electron (v${versions.electron()})`

const bt_videoSelect = document.getElementById('btnVideo')
const filePathElement = document.getElementById('videoFilePath')

bt_videoSelect.addEventListener('click', async () => {
    const videoFilePath = await window.electronAPI.selectFile()
    filePathElement.innerText = videoFilePath
    const mainVideo = document.getElementById('mainVideo')
    const videoSource = document.getElementById('mainVideoSource')
    mainVideo.pause()
    videoSource.setAttribute('src', 'file:' + videoFilePath)
    mainVideo.load()
    mainVideo.play()
})


const bt_detect = document.getElementById('bt001')
bt_detect.addEventListener('click', function(clickEvent){
   videoFileName = document.getElementById('videoFilePath').innerText
   document.getElementById('pdetect').innerHTML = 'Now detecting from ' + videoFileName
   const message = window.runpython.detect({"filename": videoFileName})
   const message2 = window.runpython.on("return_data", async(data)=>{
       document.getElementById('pdetect').innerHTML = '<pre>' + data + '</pre>'
   })
})

bt_analyze = document.getElementById('bt002')
bt_analyze.addEventListener('click', function(clickEvent){
    document.getElementById('panalyze').innerHTML = ''
   const message = window.runpython.analyze({"send_data":"analyze"})
   const message2 = window.runpython.on("return_data", async(data)=>{
       document.getElementById('panalyze').innerHTML = '<pre>' + data + '</pre>'
   })
})

bt_quit = document.getElementById('btquit')
bt_quit.addEventListener('click', function(clickEvent){
    window.close()
})
