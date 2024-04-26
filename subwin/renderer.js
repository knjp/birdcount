const information = document.getElementById('info')
information.innerText = `This app is using Chrome (v${versions.chrome()}), Node.js (v${versions.node()}), and Electron (v${versions.electron()})`

const bt_videoSelect = document.getElementById('btnVideo')
const bt_3dscatter = document.getElementById('btn3dscatter')
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

bt_3dscatter.addEventListener('click', async () => {
    const message = window.runpython.scatter3d({"send_data":"analyze"})
    const message2 = window.runpython.on("return_data", async(data)=>{
        console.log(string(data))
    })
})

const bt_close = document.getElementById('btnclose')

bt_close.addEventListener('click', function(clickEvent){
    window.close()
})