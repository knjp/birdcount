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


var ptime = 0
function printTime(){
    var now = new Date()
    var year = now.getFullYear()
    var month = now.getMonth()
    const ct = document.getElementById('ctime')
    ct.innerHTML = now.toLocaleTimeString()
    ptime = ptime + 1
    const pt = document.getElementById('ptime')
    pt.innerHTML = ptime + 's'
}

const bt_detect = document.getElementById('bt001')
const bt_analyze = document.getElementById('bt002')
const bt_quit = document.getElementById('btquit')

bt_detect.addEventListener('click', function(clickEvent){
    bt_detect.disabled = true
    bt_analyze.disabled = true
    bt_quit.disabled = true
    bt_videoSelect.disabled = true
    var processtime = setInterval(printTime, 1000)
    videoFileName = document.getElementById('videoFilePath').innerText
    document.getElementById('pdetect').innerHTML = 'Now detecting from ' + videoFileName
    const message = window.runpython.detect({"filename": videoFileName})
    const message2 = window.runpython.on("return_data", async(data)=>{
        document.getElementById('pdetect').innerHTML = '<pre>' + data + '</pre>'
        bt_detect.disabled = false
        bt_analyze.disabled = false
        bt_quit.disabled = false
        bt_videoSelect.disabled = false
        clearInterval(processtime)
    })
})

bt_analyze.addEventListener('click', function(clickEvent){
    document.getElementById('panalyze').innerHTML = ''
    const message = window.runpython.analyze({"send_data":"analyze"})
    const message2 = window.runpython.on("return_data", async(data)=>{
       document.getElementById('panalyze').innerHTML = '<pre>' + data + '</pre>'
   })
})

bt_quit.addEventListener('click', function(clickEvent){
    window.close()
})
