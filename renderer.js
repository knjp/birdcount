const information = document.getElementById('info')
information.innerText = `This app is using Chrome (v${versions.chrome()}), Node.js (v${versions.node()}), and Electron (v${versions.electron()})`

bt_detect = document.getElementById('bt001')
bt_detect.addEventListener('click', function(clickEvent){
    document.getElementById('ppython').innerHTML = ''
   const message = window.runpython.detect({"filename":"a_002.mp4"})
   const message2 = window.runpython.on("return_data", async(data)=>{
       document.getElementById('ppython').innerHTML = data
   })
})

bt_analyze = document.getElementById('bt002')
bt_analyze.addEventListener('click', function(clickEvent){
    document.getElementById('ppython').innerHTML = ''
   const message = window.runpython.analyze({"send_data":"analyze"})
   const message2 = window.runpython.on("return_data", async(data)=>{
       document.getElementById('ppython').innerHTML = data
   })
})

bt_quit = document.getElementById('btquit')
bt_quit.addEventListener('click', function(clickEvent){
    window.close()
})