const information = document.getElementById('info')
information.innerText = `This app is using Chrome (v${versions.chrome()}), Node.js (v${versions.node()}), and Electron (v${versions.electron()})`

const bt_videoSelect = document.getElementById('btnVideo')
const filePathElement = document.getElementById('videoFilePath')

function setMainVideo(videoFilePath) {
    filePathElement.innerText = videoFilePath
    const mainVideo = document.getElementById('mainVideo')
    const videoSource = document.getElementById('mainVideoSource')
    mainVideo.pause()
    videoSource.setAttribute('src', 'file:' + videoFilePath)
    mainVideo.load()
    mainVideo.play()
}

bt_videoSelect.addEventListener('click', async () => {
    const videoFilePath = await window.electronAPI.selectLoadVideoFile()
    filePathElement.innerText = videoFilePath
    const mainVideo = document.getElementById('mainVideo')
    const videoSource = document.getElementById('mainVideoSource')
    mainVideo.pause()
    videoSource.setAttribute('src', 'file:' + videoFilePath)
    mainVideo.load()
    mainVideo.play()
})

const bt_csvDownload = document.getElementById('btnCSV')
bt_csvDownload.addEventListener('click', () => {
    const saveFilePath = window.electronAPI.saveCSVFile()
    console.log(saveFilePath)
    //window.electronAPI.saveCSVFile()
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

function buttonDisable() {
    bt_detect.disabled = true
    bt_analyze.disabled = true
    bt_quit.disabled = true
    bt_videoSelect.disabled = true
}

function buttonEnable() {
    bt_detect.disabled = false
    bt_analyze.disabled = false
    bt_quit.disabled = false
    bt_videoSelect.disabled = false
}

bt_detect.addEventListener('click', function(clickEvent){
    buttonDisable()
    ptime = 0
    var processtime = setInterval(printTime, 1000)
    videoFileName = document.getElementById('videoFilePath').innerText
    document.getElementById('pdetect').innerHTML = 'Now detecting from ' + videoFileName
    const message = window.runpython.detect({"filename": videoFileName})
    const message2 = window.runpython.on("return_data", async(data)=>{
        dstrs = String(data).split(',')
        console.log('hello')
        console.log(dstrs)
        datas = dstrs[0] + '<br>' + dstrs[1] + '<br>' + dstrs[2] + '<br>' + dstrs[3]
        document.getElementById('pdetect').innerHTML = '<pre>' + datas + '</pre>'
        buttonEnable()
        clearInterval(processtime)
    })
})

bt_analyze.addEventListener('click', function(clickEvent){
    const img1 = document.getElementById('resultFig1')
    const img2 = document.getElementById('resultFig2')
    img1.setAttribute('src', '')
    img2.setAttribute('src', '')
    buttonDisable()
    ptime = 0
    var processtime = setInterval(printTime, 1000)
    document.getElementById('panalyze').innerHTML = ''
    const message = window.runpython.analyze({"send_data":"analyze"})
    const message2 = window.runpython.on("return_data", async(data)=>{
        dataStr = String(data)
        strs = dataStr.split(',')
        stats = strs[2] + '<br>' + strs[3] + '<br>'  + strs[4] + '<br>' + strs[5]
        document.getElementById('panalyze').innerHTML = '<pre>' + stats + '</pre>'
        buttonEnable()
        clearInterval(processtime)
        img1.setAttribute('src', 'yolo/resultsFig2.png?' + new Date().getTime())
        img2.setAttribute('src', 'yolo/resultsFig3.png?' + new Date().getTime())
        setMainVideo(strs[0])
   })
})

bt_quit.addEventListener('click', function(clickEvent){
    window.close()
})
