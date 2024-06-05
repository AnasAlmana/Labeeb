<script setup lang="ts">

import type { ChatMessage, CleanChatMessage } from "@/types";
import { ref, watch, nextTick, onMounted, computed } from "vue";
import { RouterLink } from "vue-router";
import { hugginggpt } from "@/api/hugginggpt";
import { chatgpt } from "@/api/chatgpt";
import Loading from "@/components/Loading.vue";
import promptCollection from "@/prompt";
import { HUGGINGGPT_BASE_URL } from "@/config";

let isChatgpt = ref(false);
let isTalking = ref(false);
let isConfig = ref<boolean>(true);
let title = ref<string>();
let mode = ref<string>("default");


title.value = "LABEEB";

isConfig.value = (isChatgpt.value)? true : false

const chatListDom = ref<HTMLDivElement>();
// const pdf = ref<HTMLDivElement>();
let messageContent = ref("");

const roleAliasChatHuggingGPT = { user: "أنت", assistant: "لَبِيب", system: "System" };
const roleAliasChatGPT = { user: "أنت", assistant: "ChatGPT", system: "System" };
const roleAlias = ref(isChatgpt.value? roleAliasChatGPT: roleAliasChatHuggingGPT);
const messageList = ref<ChatMessage[]>(isChatgpt.value? promptCollection["chatgpt"][mode.value]: promptCollection["hugginggpt"][mode.value]);

onMounted(() => {
  const apiKey = loadConfig();
  if (apiKey) {
    // switchConfigStatus(); //close
    isConfig.value = false
  }
});

async function sendChatMessage() {
  isTalking.value = true;
  const input = messageContent.value
  messageList.value.push(
    { role: "user", content: input, type: "text", first: true},
  )

  clearMessageContent();
  var clean_messages: CleanChatMessage[] = []
  for (let message of messageList.value) {
    if (message.first && message.role != "system") {
      clean_messages.push({role: message.role, content: message.content})
    }
  }
  messageList.value.push(
    { role: "assistant", content: "", type: "text", first: true},
  )
  if (isChatgpt.value) {
    var { status, data, message } = await chatgpt(clean_messages, loadConfig());
  } else {
    var { status, data, message } = await hugginggpt(clean_messages);
  }

  messageList.value.pop()
  if (status === "success" ) {
    if (data) {
      messageList.value.push(
        { role: "assistant", content: data, type: "text", first: true }
      );
    } else {
      messageList.value.push(
        { role: "assistant", content: "empty content", type: "text", first: true }
      );
    }
  } else {
    messageList.value.push(
      { role: "system", content: message, type: "text", first: true }
    );
  }
  isTalking.value = false;
}


const messageListMM = computed(() => {
  var messageListMM: ChatMessage[] = []
  for (var i = 0; i < messageList.value.length; i++) {
    var message = messageList.value[i]
    if (message.type != "text") {
      messageListMM.push(message)
      continue
    }
    var content = message.content
    var role = message.role
    
    var image_urls = content.match(/(http(s?):|\/)([/|.|\S||\w|:|-])*?\.(?:jpg|jpeg|tiff|gif|png)/g)
    var image_reg = new RegExp(/(http(s?):|\/)([/|.|\S|\w|:|-])*?\.(?:jpg|jpeg|tiff|gif|png)/g)
    
    var orig_content = content
    var seq_added_accum = 0
    if (image_urls){
      for (var j = 0; j < image_urls.length; j++) {
        // @ts-ignore
        var start = image_reg.exec(orig_content).index
        var end = start + image_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
          <a class="inline-flex text-sky-800 font-bold items-baseline" target="_blank" href="${image_urls[j].startsWith("http")?image_urls[j]:HUGGINGGPT_BASE_URL+image_urls[j]}">
              <img src="${image_urls[j].startsWith("http")?image_urls[j]:HUGGINGGPT_BASE_URL+image_urls[j]}" alt="" class="inline-flex self-center w-5 h-5 rounded-full mx-1" />
              <span class="mx-1">[Image]</span>
          </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - image_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!image_urls[j].startsWith("http")){
          image_urls[j] = HUGGINGGPT_BASE_URL + image_urls[j]
        }
      }
    }
  
    orig_content = content
    var audio_urls = content.match(/(http(s?):|\/)([/|.|\w|\S|:|-])*?\.(?:flac|wav)/g)
    var audio_reg = new RegExp(/(http(s?):|\/)([/|.|\w|\S|:|-])*?\.(?:flac|wav)/g)
  
    var seq_added_accum = 0
    if (audio_urls){
      for (var j = 0; j < audio_urls.length; j++) {
        // @ts-ignore
        var start = audio_reg.exec(orig_content).index
        var end = start + audio_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
            <a class="text-sky-800 inline-flex font-bold items-baseline" target="_blank" href="${audio_urls[j].startsWith("http")?audio_urls[j]:HUGGINGGPT_BASE_URL+audio_urls[j]}">
              <img class="inline-flex self-center w-5 h-5 rounded-full mx-1" src="/audio.svg"/>
              <span class="mx-1">[Audio]</span>
            </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - audio_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!audio_urls[j].startsWith("http")){
          audio_urls[j] = HUGGINGGPT_BASE_URL + audio_urls[j]
        }
      }
    }

    orig_content = content
    var video_urls = content.match(/(http(s?):|\/)([/|.|\w|\s|:|-])*?\.(?:mp4)/g)
    var video_reg = new RegExp(/(http(s?):|\/)([/|.|\w|\s|:|-])*?\.(?:mp4)/g)
  
    var seq_added_accum = 0
    if (video_urls){
      for (var j = 0; j < video_urls.length; j++) {
        // @ts-ignore
        var start = video_reg.exec(orig_content).index
        var end = start + video_urls[j].length
        start += seq_added_accum
        end += seq_added_accum
        const replace_str = `<span class="inline-flex items-baseline">
            <a class="text-sky-800 inline-flex font-bold items-baseline" target="_blank" href="${video_urls[j].startsWith("http")?video_urls[j]:HUGGINGGPT_BASE_URL+video_urls[j]}">
              <img class="inline-flex self-center w-5 h-5 rounded-full mx-1" src="/video.svg"/>
              <span class="mx-1">[video]</span>
            </a>
          </span>`
        const rep_length = replace_str.length
        seq_added_accum += (rep_length - video_urls[j].length)
        content = content.slice(0, start) + replace_str + content.slice(end)
        
        if(!video_urls[j].startsWith("http")){
          video_urls[j] = HUGGINGGPT_BASE_URL + video_urls[j]
        }
      }
    }

    message = {role: role, content: content, type: "text", first: true}
    messageListMM.push(message)
    // de-depulicate
    // @ts-ignore
    image_urls = [...new Set(image_urls)]
    // @ts-ignore
    audio_urls = [...new Set(audio_urls)]
    // @ts-ignore
    video_urls = [...new Set(video_urls)]
    if (image_urls) {
      
      for (var j = 0; j < image_urls.length; j++) {
        messageListMM.push({role: role, content: image_urls[j], type: "image", first: false})
      }
    }
    if (audio_urls) {
      for (var j = 0; j < audio_urls.length; j++) {
        messageListMM.push({role: role, content: audio_urls[j], type: "audio", first: false})
      }
    }
    if (video_urls) {
      for (var j = 0; j < video_urls.length; j++) {
        messageListMM.push({role: role, content: video_urls[j], type: "video", first: false})
      }
    }
    // if (code_blocks){
    //   for (var j = 0; j < code_blocks.length; j++) {
    //     messageListMM.push({role: role, content: code_blocks[j], type: "code", first: false})
    //   }
    // }
  }
  // nextTick(()=>scrollToBottom())
  return messageListMM
})

const sendOrSave = () => {
  if (!messageContent.value.length) return;
  if (isConfig.value) {
    if (saveConfig(messageContent.value.trim())) {
      switchConfigStatus();
    }
    clearMessageContent();
  } else {
    sendChatMessage();
  }
};

const clickConfig = () => {
  if (!isConfig.value) {
    messageContent.value = loadConfig();
  } else {
    clearMessageContent();
  }
  switchConfigStatus();
};


function saveConfig(apiKey: string) {
  if (apiKey.slice(0, 3) !== "sk-" || apiKey.length !== 51) {
    alert("Illegal API Key");
    return false;
  }
  localStorage.setItem("apiKey", apiKey);
  return true;
}

function loadConfig2() {
  return localStorage.getItem("apiKey") ?? "";
}

function loadConfig() {
  return localStorage.getItem("apiKey") ?? "";
}

function scrollToBottom() {
  if (!chatListDom.value) return;
  // scrollTo(0, chatListDom.value.scrollHeight);
  chatListDom.value.scrollIntoView(false);
}

function switchConfigStatus() {
  isConfig.value = !isConfig.value;
}

function clearMessageContent() {
  messageContent.value = "";
}




// const generateScreenshot = async ()=>{
//   const canvas = await html2canvas(pdf.value)
//   let a = new jsPDF("p", "mm", "a4")
//   // 
//   a.addImage(canvas.toDataURL("image/png"), "PNG", 0, 0, 211, 298);
//   a.save("screenshot.pdf")
// }

watch(mode, ()=> {
  if (isChatgpt.value) {
    messageList.value = promptCollection["chatgpt"][mode.value]
  } else {
    messageList.value = promptCollection["hugginggpt"][mode.value]
  }
})

watch(isChatgpt, () => {
  if (isChatgpt.value) {
    mode.value = "default"
    messageList.value = promptCollection["chatgpt"]["default"]
  } else {
    mode.value = "default"
    messageList.value = promptCollection["hugginggpt"]["default"]
  }
});

// messageList -> messageListMM
watch(messageListMM, () => nextTick(() => {
  nextTick(()=>scrollToBottom())
  }));

  async function playAudio() {
  try {
    const response = await fetch("/home/ubuntu/labeeb/JARVIS/hugginggpt/server/responses/output.mp3");
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  } catch (error) {
    console.error('Error fetching and playing audio:', error);
  }
}
let recorder: MediaRecorder;
let audioChunks: Blob[] = [];
let statusMessage = ref();
const isRecordingModalOpen = ref(false);
let isRecording = false;
let silenceTimeout: number | null = null;
const SILENCE_THRESHOLD = 8; // Adjust this value to change sensitivity
const SILENCE_DETECTION_INTERVAL = 0; // Interval for silence detection in milliseconds
const SILENCE_DURATION = 1500;
let audio = new Audio();

async function startRecording() {

  console.log("hi")
  
  isRecordingModalOpen.value = true
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    statusMessage.value = 'يمكنك البدء في التحدث...';

    recorder = new MediaRecorder(stream);
    audioChunks = [];

    recorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };

    recorder.onstop = async () => {
      statusMessage.value = 'توقف التسجيل';
      let audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });

      stream.getTracks().forEach(track => track.stop());
    

      const formData = new FormData();
      formData.append('audio', audioBlob, 'input.mp3');

      const response = await fetch('http://127.0.0.1:5000/voice', {
        method: 'POST',
        body: formData
      });

      audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audio) {
            audio.pause();
            audio.currentTime = 0;
        }

      audio = new Audio(audioUrl);

      await playAudioAndWait(audio);

      if (isRecordingModalOpen.value = false){
        stopAudio()

      } else{startRecording();     
                        }// This line will execute after the audio finishes playing
    };

    const AudioContext = window.AudioContext ;
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048; // Ensure FFT size is appropriate for silence detection
    const dataArray = new Uint8Array(analyser.fftSize);
    source.connect(analyser);

    function detectSilence() {
      analyser.getByteTimeDomainData(dataArray);
      const isSilent = dataArray.every(value => Math.abs(value - 128) < SILENCE_THRESHOLD); // Adjusted silence detection

      console.log('Silence detected:', isSilent); // Log silence detection status

      if (isSilent && isRecording) {
        if (silenceTimeout === null) {
          silenceTimeout = window.setTimeout(stopRecording, SILENCE_DURATION); // Stop recording after 1.5 seconds of silence
        }
      } else if (!isSilent && isRecording) {
        if (silenceTimeout !== null) {
          clearTimeout(silenceTimeout);
          silenceTimeout = null;
        }
      } else if (!isSilent && !isRecording) {
        startMediaRecorder();
      }
    }

    function startMediaRecorder() {
      recorder.start();
      isRecording = true;
      statusMessage.value = 'يسجل...';
      console.log('Recording started');
    }

    function stopRecording() {
      if (isRecording) {
        recorder.stop();
        isRecording = false;
        silenceTimeout = null;
        console.log('Recording stopped due to silence');
      }
    }

    setInterval(detectSilence, SILENCE_DETECTION_INTERVAL);

  } catch (error) {
    console.error('Error accessing microphone:', error);
    alert('Could not access the microphone. Please check permissions.');
    statusMessage.value = 'Error: Microphone access denied';
  }
}

function playAudioAndWait(audio) {
    statusMessage.value = "إجابة لبيب..."
    return new Promise((resolve) => {
        audio.addEventListener('ended', resolve, { once: true });
        audio.play();
    });
}
function stopAudio() {
    if (audio) {
        audio.pause();
        audio.currentTime = 0;
    }
}

function closeModal() {
  stopAudio()
  isRecordingModalOpen.value = false;
  statusMessage.value = 'يمكنك البدء في التحدث...';
}
async function startVoice() {
  isRecordingModalOpen.value = true
  while (isRecordingModalOpen.value == true) {
    await startRecording();
  }
  
}
function uploadImage(event: Event) {
  const target = event.target as HTMLInputElement;
  const files = target.files;
  if (files) {
    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          uploadedImages.value.push(e.target.result as string);
          messageList.value.push({
            role: "user",
            content: e.target.result as string,
            type: "image",
            first: true
          });
        }
      };
      reader.readAsDataURL(files[i]);
    }
  }
}

function triggerFileInput() {
  document.getElementById('imageInput')!.click();
}





</script>

<template>
  <div class="flex flex-row justify-center overflow-auto" dir="rtl">
    <!-- Modal for Recording -->
    <div v-if="isRecordingModalOpen" class="modal">
      <div class="modal-content">
        <span class="close" @click="closeModal" >&times;</span>
        <h1 id="status">{{ statusMessage }}</h1>
      </div>
    </div>

    <!-- Your existing HTML code -->

    <div class="flex flex-col w-full h-screen max-w-lg border-x-2 border-slate-200">
      <div class="flex flex-col w-full h-20">
        <div class="flex flex-nowrap fixed max-w-lg w-full items-center justify-center top-0 px-6 py-6 bg-gray-100 z-50 h-20">
          <img src="@/assets/logo.svg" class="w-24 ml-1 inline" />
        </div>
      </div>
      <div class="flex flex-col">
        <div class="flex flex-nowrap fixed max-w-lg w-full items-center justify-end top-0 px-6 py-6 z-50 h-20">
          <!-- Image Button -->
          <button class="bg-transparent" @click="startRecording()">
            <img src="call4.png" alt="Button Image" class="w-7 h-10">
          </button>
        </div>
      </div>
      
      <div class="flex-1 overflow-auto" ref="pdf">
        <div class="m-5" ref="chatListDom">
          <div class="relative border-2 rounded-xl p-3" 
               :class="{'bg-violet-50':item.role=='user', 'bg-blue-50':item.role=='assistant', 'bg-yellow-50':item.role=='system', 'mt-4': item.first, 'mt-1': !item.first }" 
               v-for="item of messageListMM">
            <svg xmlns="http://www.w3.org/2000/svg" v-if="!item.first"  fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 absolute -top-4 left-4 stroke-slate-400">
              <path stroke-linecap="round" stroke-linejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" />
            </svg>
            <div v-if="item.first" class="font-bold text-sm mb-3 inline">{{roleAlias[item.role]}} :</div>
            <span class="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed" v-if="item.content && item.type === 'text'">
              <div class="break-words" v-html="item.content"></div>
            </span>
            <img class="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed" v-else-if="item.content && item.type === 'image'" :src="item.content" />
            <audio controls class="w-full text-blue-100" v-else-if="item.content && item.type === 'audio'" :src="item.content"></audio>
            <video class="w-full" v-else-if="item.content && item.type === 'video'" controls>
              <source :src="item.content" type="video/mp4">
            </video>
            <pre class="" v-else-if="item.content && item.type === 'code'">
              <code>{{item.content}}</code>
            </pre>
            <Loading class="mt-2" v-else />
          </div>
        </div>
      </div>

      <div class="sticky bottom-0 w-full p-3 bg-gray-100">
        <div class="-mt-2 m-1 text-sm text-gray-500" v-if="isConfig">
          Please input OpenAI key:
        </div>
        <div class="flex">
          <textarea
            rows="2"
            style="resize:none"
            class="input"
            type="text"
            :placeholder="isConfig ? 'sk-xxxxxxxxxx' : 'ادخل رسالتك هنا...'"
            v-model="messageContent"
            @keydown.enter.prevent="isTalking || sendOrSave()">
          </textarea>

          <div class="flex flex-col justify-center">
            <button class="btn bg-purple-700 hover:bg-purple-800 disabled:bg-purple-400 focus:bg-purple-800 text-sm w-20 m-1 h-7 p-1" 
                    :disabled="!messageList[messageList.length - 1].content"
                    @click="sendOrSave">
              {{ isConfig ? "حفظ" : "ارسال" }}
            </button>
            <button class="btn bg-gray-700 hover:bg-gray-800 disabled:bg-gray-400 focus:bg-gray-800 text-sm w-20 m-1 h-7 p-1 flex items-center justify-center"
            @click="triggerFileInput">
              <img src="image.png" alt="Button Image" class="w-5 h-5 justify-center item-center" >
            </button>
            <input type="file" id="imageInput" accept="image/*" multiple style="display:none;" @change="uploadImage">
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<!-- Add modal styles -->
<style scoped>
.modal {
  display: flex;
  justify-content: center;
  align-items: center;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 400px;
  text-align: center;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
pre {
  font-family: -apple-system, "Noto Sans", "Helvetica Neue", Helvetica,
    "Nimbus Sans L", Arial, "Liberation Sans", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
    "Microsoft YaHei", "Wenquanyi Micro Hei", "WenQuanYi Zen Hei", "ST Heiti",
    SimHei, "WenQuanYi Zen Hei Sharp", sans-serif;
}
audio {
  width: 100%;
  background-color: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 0.25rem;
  padding: 0.25rem;
  margin: 0;
}
::-webkit-scrollbar {
  display: none;
}
</style>

