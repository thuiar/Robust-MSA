<template>
  <div class="appContainer">
    <div class="stepTop">
      <div class="arrayTitle" :style="{'width':arrayTitle+'px'}">Robust-MSA</div>
      <el-steps :active="activeStep" class="elSteps" simple>
        <el-step title="Upload Video" :icon="activeStep>0?CircleCheckFilled:UploadFilled" />
        <el-step title="Revise Transcript" :icon="activeStep>1?CircleCheckFilled:List" />
        <el-step title="Modify Video" :icon="activeStep>2?CircleCheckFilled:EditPen" />
        <el-step title="Select Methods" :icon="activeStep>3?CircleCheckFilled:Operation" />
        <el-step title="View Results" :icon="activeStep>4?CircleCheckFilled:Histogram" />
      </el-steps>
    </div>
    <div class="firstPageContainer" v-if="activeStep===0">
      <div
        class="tipNote"
      >Note: Uploaded file should be no more than 10MB. Currently only support English.</div>
      <el-upload
        class="upload-demo"
        ref="upload"
        action="#"
        :auto-upload="false"
        :limit="1"
        drag
        :on-change="uploadMp4"
        accept=".MP4, .AVI, .MOV, .MKV, .3GP, .M4V, .FLV, .MPG, .mp4, .avi, .mov, .mkv, .3gp, .m4v, .flv, .mpg"
        :show-file-list="false"
      >
        <div v-if="typeof currentFile == 'undefined' || currentFile == ''">
          <el-icon class="el-icon--upload">
            <upload-filled />
          </el-icon>
          <div class="el-upload__text">
            Drop file here or
            <em>click to upload</em>
          </div>
        </div>
        <div v-else class="successUpload">
          <el-icon class="el-icon--upload">
            <upload-filled />
          </el-icon>
          <div class="el-upload__text">
            Drop file here or
            <em>click to upload</em>
          </div>
        </div>
        <template #tip>
          <div
            class="el-upload__tip"
            v-if="!(typeof currentFile == 'undefined' || currentFile == '')"
          >
            <el-icon>
              <VideoCameraFilled />
            </el-icon>
            <span>Filename: {{currentFile.name}}</span>
          </div>
        </template>
      </el-upload>
      <el-button
        class="aligned elButton nextbutton"
        @click="nextButton(1)"
        size="small"
        type="primary"
      >Next</el-button>
    </div>
    <div class="firstPageContainer" v-if="activeStep===1">
      <video id="my-player1" class="video-js1" controls data-setup="{}">
        <source :src="McurrentVideoUrl" type="video/mp4" />
      </video>
      <div class="transcriptCon">
        <div class="transcriptTip">Transcript:</div>
        <el-input type="textarea" :rows="2" placeholder="Please enter" v-model="textarea"></el-input>
      </div>
      <div>
        <el-button
          class="aligned elButton nextbutton"
          @click="nextButton(2)"
          size="small"
          type="primary"
        >Next</el-button>
      </div>
    </div>
    <div class="pageContainer" v-else-if="activeStep===2">
      <div class="modifiedCon">
        <div class="modifiedTopContainer">
          <div class="previewContainer">
            <div class="previewTop">
              <div class="topTip">Preview</div>
              <div class="topDescribe">Click "generate" button to preview changes.</div>
            </div>
            <div id="modifiedContainer">
              <video id="modified-my-player" class="video-js vjs-big-play-centered" controls>
                <source :src="McurrentVideoUrl" type="video/mp4" />
              </video>
            </div>
          </div>
          <div class="methodsContainer">
            <div class="methodsTop">
              <div class="topTip">Modify Methods</div>
              <div class="topDescribe">
                <span>Drag & drop</span> methods onto word to take effect
              </div>
            </div>
            <el-affix :offset="affixStatus?20:-10000" style="marginTop:15px">
              <el-card shadow="always" class="methodContent">
                <svg
                  t="1661607155279"
                  v-if="affixStatus"
                  @click="affixStatus=!affixStatus"
                  class="fixedIcon"
                  viewBox="0 0 1024 1024"
                  version="1.1"
                  xmlns="http://www.w3.org/2000/svg"
                  p-id="3796"
                  width="128"
                  height="128"
                >
                  <path
                    d="M381.298 418.828h-157.703l-37.575 38.272 155.61 158.377-278.212 345.128 356.040-265.838 154.71 157.41 38.813-39.51 2.407-157.972 238.838-313.29 71.685 73.013 34.695-35.28-310.185-315.743-34.672 35.257 77.287 79.402-311.737 240.773z"
                    p-id="3797"
                    fill="#e16531"
                  />
                </svg>
                <svg
                  t="1661607155279"
                  class="fixedIcon"
                  v-else
                  @click="affixStatus=!affixStatus"
                  viewBox="0 0 1024 1024"
                  version="1.1"
                  xmlns="http://www.w3.org/2000/svg"
                  p-id="3796"
                  width="128"
                  height="128"
                >
                  <path
                    d="M381.298 418.828h-157.703l-37.575 38.272 155.61 158.377-278.212 345.128 356.040-265.838 154.71 157.41 38.813-39.51 2.407-157.972 238.838-313.29 71.685 73.013 34.695-35.28-310.185-315.743-34.672 35.257 77.287 79.402-311.737 240.773z"
                    p-id="3797"
                    fill="#707070"
                  />
                </svg>
                <div class="methodContainer">
                  <div class="modalityMethod">
                    <el-icon v-if="dragDisplay.text">
                      <ArrowDown />
                    </el-icon>
                    <el-icon v-else>
                      <ArrowRight />
                    </el-icon>
                    <span @click="dragDisplay.text=!dragDisplay.text">Text Methods</span>
                  </div>
                  <div v-show="dragDisplay.text">
                    <div class="methodState">
                      <span draggable="true" @dragend="dragEnd($event,'t','replace','off')">Replace</span>
                    </div>
                    <div class="methodState">
                      <span draggable="true" @dragend="dragEnd($event,'t','remove','')">Remove</span>
                    </div>
                  </div>
                  <div class="modalityMethod">
                    <el-icon v-if="dragDisplay.video">
                      <ArrowDown />
                    </el-icon>
                    <el-icon v-else>
                      <ArrowRight />
                    </el-icon>
                    <span @click="dragDisplay.video=!dragDisplay.video">Video Methods</span>
                  </div>
                  <div v-show="dragDisplay.video">
                    <div class="methodState">
                      <span draggable="true" @dragend="dragEnd($event,'v','blank','')">Blank</span>
                    </div>
                    <div class="methodState">
                      <span draggable="true" @dragend="dragEnd($event,'v','gblur',gblurValue)">Gblur</span>
                      <el-select
                        v-model="gblurValue"
                        size="small"
                        class="gblurSelect"
                        placeholder="Select"
                      >
                        <el-option
                          v-for="item in gblurOptions"
                          :key="item.value"
                          :label="item.label"
                          :value="item.value"
                        />
                      </el-select>
                    </div>
                  </div>
                  <div class="modalityMethod">
                    <el-icon v-if="dragDisplay.audio">
                      <ArrowDown />
                    </el-icon>
                    <el-icon v-else>
                      <ArrowRight />
                    </el-icon>
                    <span @click="dragDisplay.audio=!dragDisplay.audio">Audio Methods</span>
                  </div>
                  <div v-show="dragDisplay.audio">
                    <div class="methodState">
                      <span draggable="true" @dragend="dragEnd($event,'a','mute','')">Mute</span>
                    </div>
                    <div class="methodStateNoise">
                      <!-- <span
                        draggable="true"
                        @dragend="dragEnd($event,'a','noise-white','low')"
                      >Noise-white</span>-->
                      <div class="noiseTitle" @click="dragDisplay.noise=!dragDisplay.noise">
                        <el-icon v-if="dragDisplay.noise">
                          <ArrowDown />
                        </el-icon>
                        <el-icon v-else>
                          <ArrowRight />
                        </el-icon>

                        <el-tooltip
                          class="box-item"
                          effect="dark"
                          content="Apply to whole audio"
                          placement="right"
                        >
                          <span>Noise</span>
                        </el-tooltip>
                      </div>
                      <div class="noiseContainer" v-show="dragDisplay.noise">
                        <div class="noiseItem">
                          <span>Noise white</span>
                          <el-slider
                            class="elSlider"
                            @input="sliderInput"
                            :min="0"
                            :max="1"
                            :step="0.01"
                            v-model="noiseObj.white"
                            size="small"
                          />
                        </div>
                        <div class="noiseItem">
                          <span>Noise metro</span>
                          <el-slider
                            class="elSlider"
                            @input="sliderInput"
                            :min="0"
                            :max="1"
                            :step="0.01"
                            v-model="noiseObj.metro"
                            size="small"
                          />
                        </div>
                        <div>
                          <div class="noiseItem">
                            <span>Noise office</span>
                            <el-slider
                              class="elSlider"
                              @input="sliderInput"
                              :min="0"
                              :max="1"
                              :step="0.01"
                              v-model="noiseObj.office"
                              size="small"
                            />
                          </div>
                        </div>
                        <div>
                          <div class="noiseItem">
                            <span>Noise park</span>
                            <el-slider
                              class="elSlider"
                              @input="sliderInput"
                              :min="0"
                              :max="1"
                              :step="0.01"
                              v-model="noiseObj.park"
                              size="small"
                            />
                          </div>
                        </div>
                        <div>
                          <div class="noiseItem">
                            <span>Noise diner</span>
                            <el-slider
                              class="elSlider"
                              @input="sliderInput"
                              :min="0"
                              :max="1"
                              :step="0.01"
                              v-model="noiseObj.diner"
                              size="small"
                            />
                          </div>
                        </div>
                        <div>
                          <div class="noiseItem">
                            <span>Noise traffic</span>
                            <el-slider
                              class="elSlider"
                              @input="sliderInput"
                              :min="0"
                              :max="1"
                              :step="0.01"
                              v-model="noiseObj.traffic"
                              size="small"
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </el-card>
            </el-affix>
          </div>
        </div>
        <div class="modifiedWaveform" v-if="modifiedWaveformSwitch">
          <div id="modifiedWaveTimeline" ref="modifiedWaveTimeline"></div>
          <div
            id="modifiedWaveform"
            ref="modifiedWaveform"
            @click="seekToTime(null)"
            @contextmenu.prevent="rightClickItem($event,'wave')"
            @mouseover="waveOver($event,'rgba(255, 228, 196, 0.5)','rgba(255, 228, 196, 0.5)')"
            @mouseout="waveOver($event,'rgba(181, 198, 241, 0.2)','#fff')"
          ></div>
        </div>
        <div class="textContent">
          <span
            @mouseenter="textOver($event,index, 'rgba(255, 228, 196, 0.5)','rgba(255, 228, 196, 0.5)')"
            @mouseleave="textOver($event,index, 'rgba(181, 198, 241, 0.2)','#fff')"
            @contextmenu.prevent="rightClickItem($event,'text')"
            @dragover="dragOver"
            @click="seekToTime(textitem.start)"
            v-for="(textitem,index) in modifiedTextList"
            :key="textitem"
            :id="index"
          >{{textitem.text}}</span>
        </div>
        <Legend />
        <el-button class="aligned elButton generate" @click="generate" type="danger">Generate</el-button>
        <el-button
          class="aligned elButton nextbutton"
          :type="McurrentVideoUrl== currentVideoUrl?'info':'primary'"
          @click="nextButton(3)"
        >Next</el-button>
      </div>
    </div>
    <div class="pageContainer" style="position: relative;" v-else-if="activeStep===3">
      <select-method class="selectMethod" @transmitMethods="methodList = $event" />
      <el-button
        class="aligned elButton nextbutton"
        style="marginTop:60px"
        @click="nextButton(4)"
        type="primary"
      >Next</el-button>
    </div>
    <div class="pageContainer" v-else-if="activeStep===4">
      <view-result
        :textList="modifiedTextList"
        :videoUrl="McurrentVideoUrl"
        :defenceVideoUrl="defenceVideoUrl"
        :originalVideoUrl="currentVideoUrl"
        :duration="duration"
        :editAligned="editAligned"
        :viewResults="viewResults"
        :methodData="methodData"
      />
    </div>
    <div
      id="popup"
      :style="{ top: showPositionTop + 'px', left: showPositionLeft + 'px',position:'absolute' }"
      @mouseleave="(modalityDisplay.audio=false,modalityDisplay.gblur=false,modalityDisplay.video=false,modalityDisplay.text=false)"
    >
      <div v-show="rightPopup" :style="{position:'absolute'}" class="popupContainer">
        <div
          class="modality"
          @mouseover="(modalityDisplay.audio=false,modalityDisplay.gblur=false,modalityDisplay.video=true,modalityDisplay.text=false)"
        >
          <span>Video</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modality"
          @mouseover="(modalityDisplay.audio=true,modalityDisplay.gblur=false,modalityDisplay.video=false,modalityDisplay.text=false)"
        >
          <span>Audio</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modality"
          @mouseover="(modalityDisplay.audio=false,modalityDisplay.gblur=false,modalityDisplay.video=false,modalityDisplay.text=true)"
        >
          <span>Text</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modalityClose"
          v-show="modalityDisplay.clear"
          @click="modalityModify('clear','all')"
          @mouseover="(modalityDisplay.audio=false,modalityDisplay.gblur=false,modalityDisplay.video=false,modalityDisplay.text=false)"
        >Clear</div>
      </div>
      <div
        v-show="rightPopup && (modalityDisplay.video || modalityDisplay.audio || modalityDisplay.text)"
        :style="{ top: (modalityDisplay.audio?50:0)+(modalityDisplay.video?10:0)+(modalityDisplay.text?90:0)+ 'px', left: 140+ 'px',position:'relative' }"
        class="popupContainer"
      >
        <div v-show="modalityDisplay.video">
          <div
            class="state"
            @mouseover="modalityDisplay.gblur=false"
            @click="modalityModify('blank','v')"
          >
            <span>Blank</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @mouseover="modalityDisplay.gblur=true">
            <span>Gblur</span>
            <el-icon>
              <ArrowRight />
            </el-icon>
          </div>
        </div>
        <div v-show="modalityDisplay.audio">
          <div class="state" @click="modalityModify('mute','a')">
            <span>Mute</span>
            <el-icon></el-icon>
          </div>
          <!-- <div class="state" @click="modalityModify('noise-white','a')">Noise-white</div> -->
        </div>
        <div v-show="modalityDisplay.text">
          <div class="state" @click="modalityModify('replace','t')">
            <span>Replace</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('remove','t')">
            <span>Remove</span>
            <el-icon></el-icon>
          </div>
        </div>
      </div>
      <div
        v-show="rightPopup && modalityDisplay.gblur "
        :style="{ top:-30 + 'px', left: 280+ 'px',position:'relative' }"
        class="popupContainer"
      >
        <div>
          <div class="state" @click="modalityModify('gblur','v','low')">
            <span style="display:block;width:70px">Low</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('gblur','v','medium')">
            <span style="display:block;width:70px">Medium</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('gblur','v','high')">
            <span style="display:block;width:70px">High</span>
            <el-icon></el-icon>
          </div>
        </div>
      </div>
    </div>
    <el-dialog v-model="dialogReplaceText" title="Text replacement" width="30%">
      <div class="dialogReplaceTextContainer">
        <div class="dialogReplaceItem">
          <div class="dialogReplaceTitle">Word:</div>
          <el-input readonly v-model="dialogReplaceTitle" />
        </div>
        <div class="dialogReplaceItem">
          <div class="dialogReplaceTitle">Alternate:</div>
          <el-input v-model="alternateText" placeholder="Please input" />
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button type="primary" @click="dialogConfirm">Confirm</el-button>
        </span>
      </template>
    </el-dialog>
    <div
      class="ztooltip"
      v-if="overTooltip.replace||overTooltip.remove||overTooltip.blank||overTooltip.gblur||overTooltip.mute||overTooltip.noise_white"
      :style="{
      top:tooltipTop+'px',
      left:tooltipLeft+'px'
    }"
    >
      <span v-if="overTooltip.replace">Text: Replace</span>
      <span v-if="overTooltip.remove">Text: Remove</span>
      <span v-if="overTooltip.blank">Video: Blank</span>
      <span v-if="overTooltip.gblur">Video: Gblur</span>
      <span v-if="overTooltip.mute">Audio: Mute</span>
      <span v-if="overTooltip.noise_white">Audio: Noise-white</span>
    </div>
    <div
      class="triangle"
      v-if="overTooltip.replace||overTooltip.remove||overTooltip.blank||overTooltip.gblur||overTooltip.mute||overTooltip.noise_white"
      :style="{
      top:triangleTop+'px',
      left:triangleLeft+'px'
    }"
    ></div>
  </div>
</template>

<script setup>
import { ref, inject, watchEffect, computed, onMounted } from "vue";
import {
  EditPen,
  List,
  UploadFilled,
  Operation,
  Histogram,
  CircleCheckFilled,
  Flag
} from "@element-plus/icons-vue";
import Legend from "@/components/Legend.vue";
import SelectMethod from "@/components/SelectMethod.vue";
import ViewResult from "@/components/ViewResult.vue";
import { uploadVideo, uploadTranscript } from "@/api/upload";
import OperationMethod from "@/utils/operation.js";
import { callASR } from "@/api/detection";
import { videoEditAligned, getFileFromUrl } from "@/api/modify";
import { ElLoading, ElMessage } from "element-plus";
import { runMSAAligned } from "@/api/detection";
import axios from "axios";
const currentFile = ref("");
const McurrentFile = ref("");
const currentVideoUrl = ref("");
const McurrentVideoUrl = ref("");
const activeStep = ref(0);
const videoTime = ref(0);
const modifiedVideoTime = ref(0);
const modifiedWavesurfer = ref(null);
const duration = ref(0);
const originalID = ref("");
const originalPath = ref("");
const textarea = ref("");
const arrayTitle = ref("");
const showPositionTop = ref(0);
const showPositionLeft = ref(0);
const affixStatus = ref(true);
const dialogReplaceTitle = ref("");
const alternateText = ref("");
const rightPopup = ref(false);
const editAlignedIndex = ref(-1);
const methodList = ref({
  defence: [],
  models: ["tfn", "lmf", "bert_mag", "misa", "self_mm", "tfr_net", "mmim","niat"]
});
const overTooltip = ref({
  replace: false,
  remove: false,
  blank: false,
  gblur: false,
  mute: false,
  noise_white: false
});
const modalityDisplay = ref({
  video: false,
  audio: false,
  text: false,
  gblur: false,
  clear: false
});
const dragDisplay = ref({
  video: true,
  audio: true,
  text: true,
  noise: true
});
const gblurValue = ref("low");
const gblurOptions = ref([
  {
    value: "low",
    label: "Low"
  },
  {
    value: "medium",
    label: "Medium"
  },
  {
    value: "high",
    label: "High"
  }
]);
const player = ref(null);
const modifiedPlayer = ref(null);
const waveform = ref(null);
const modifiedWaveform = ref(null);
const noiseObj = ref({
  white: 0,
  metro: 0,
  office: 0,
  park: 0,
  diner: 0,
  traffic: 0
});
const upload = ref("");
const checked = ref("");
const dialogReplaceText = ref(false);
const triangleTop = ref(0);
const triangleLeft = ref(0);
const tooltipTop = ref(0);
const tooltipLeft = ref(0);
const viewResults = ref(null);
const methodData = ref(null);
const modifiedTextList = ref([]);
const editAligned = ref([]);

const defenceVideoUrl = ref("");
const regions = computed(() => {
  let dataList = [];
  modifiedTextList.value.forEach(item => {
    let itemdata = {
      start: 0,
      end: 0,
      attributes: {
        label: ""
      },
      data: {
        note: ""
      },
      loop: false,
      drag: false,
      resize: false,
      color: "rgba(181, 198, 241, 0.2)",
      handleStyle: false
    };
    itemdata.start = item.start;
    itemdata.end = item.end;
    dataList.push(itemdata);
  });
  return dataList;
});
const modifiedWaveformSwitch = ref(true);
const addVideoModified = files => {
  McurrentFile.value = files;
};
const clearModified = () => {
  McurrentFile.value = "";
};
const uploadMp4 = async (file, fileList) => {
  if (file.status === "ready") {
    const is10M = file.size / 1024 / 1024 < 10;
    if (is10M) {
      currentFile.value = file.raw;
    } else {
      ElMessage({
        message: "Uploaded file should be no more than 10MB.",
        type: "warning"
      });
    }
    upload.value.clearFiles();
  }
};
const modifiedEvent = () => {
  modifiedWavesurfer.value = OperationMethod.waveformCreate(
    modifiedWaveform.value,
    regions.value
  );
  modifiedWavesurfer.value.load(McurrentVideoUrl.value);
  modifiedWavesurfer.value.on("ready", () => {
    OperationMethod.dragBackgroundColor(
      editAligned.value,
      modifiedTextList.value.length
    );
  });
};
const nextButton = async index => {
  if (index == 1) {
    if (typeof currentFile.value == "undefined" || currentFile.value == "") {
      ElMessage({
        message: "Please select a file.",
        type: "warning"
      });
      return;
    }
    let result = await uploadFile(currentFile.value);
    if (result.code === 200) {
      // McurrentVideoUrl.value = `${window.static_url}/${result.id}/raw_video.mp4`;
      try {
          var response = await getFileFromUrl(
                `${result.id}/raw_video.mp4`,
                "raw_video.mp4"
          );
              // this.McurrentVideoUrl = `${window.static_url}/${result.id}/raw_video.mp4`
          McurrentVideoUrl.value = URL.createObjectURL(response)
      } catch (error) {
              return;
      }
      originalID.value = result.id;
      originalPath.value = result.path;
      let ASRResult = await ASRClick(result.id);
      if (ASRResult.code != 200) {
        return;
      }
      try {
        var response = await getFileFromUrl(
          `${result.id}/raw_video.mp4`,
          "raw_video.mp4"
        );
      } catch (error) {
        return;
      }
      duration.value = (await OperationMethod.getVideoDuration(response)) / 1000000;
      addVideoModified([response]);
    } else {
      return;
    }
    // addVideoModified([currentFile.value]);
    // duration.value = 12.3;
  } else if (index == 2) {
    if (textarea.value) {
      if (/\d/.test(textarea.value)) {
        ElMessage({
          message: "Please substitute numbers with text",
          type: "warning"
        });
        return;
      }
      const loading = ElLoading.service({
        lock: true,
        text: "Aligning video with transcript..."
      });
      try {
        var result = await uploadTranscript({
          videoID: originalID.value,
          transcript: textarea.value
        });
        loading.close();
      } catch (error) {
        loading.close();
        return;
      }
      modifiedTextList.value = result.data.align;
      currentVideoUrl.value = McurrentVideoUrl.value;
    }
    setTimeout(() => {
      modifiedEvent();
    }, 5);
  } else if (index == 3) {
    if (McurrentVideoUrl.value === currentVideoUrl.value) {
      ElMessage({
        message: "After modifying the video, please click generate.",
        type: "warning"
      });
      return;
    }
  } else if (index == 4) {
    if (methodList.value.models.length == 0) {
      ElMessage({
        message: "Please select at least one msa model.",
        type: "warning"
      });
      return;
    } else {
      methodData.value = methodList.value;
      methodData.value["videoID"] = originalID.value;
      const loading = ElLoading.service({
        lock: true,
        text: "Processing, please wait..."
      });
      try {
        let result = await runMSAAligned(methodData.value);
        defenceVideoUrl.value = await getFileFromUrl(
          `${originalID.value}/defended_video.mp4`,
          "defended_video.mp4"
        );
        defenceVideoUrl.value = URL.createObjectURL(defenceVideoUrl.value);
        // defenceVideoUrl.value = `${window.static_url}/${originalID.value}/defended_video.mp4`;
        viewResults.value = result.data;
        loading.close();
      } catch (error) {
        loading.close();
        return;
      }
    }
  }
  activeStep.value++;
};

const uploadFile = async file => {
  const loading = ElLoading.service({
    lock: true,
    text: "Video uploading, please wait..."
  });
  let data = new FormData();
  data.append("video", file);
  try {
    var result = await uploadVideo(data);
  } catch (error) {
    return { code: 400 };
  } finally {
    loading.close();
  }
  return result.data;
};

const clearVideo = () => {
  clearModified();
  originalID.value = "";
  originalPath.value = "";
  currentFile.value = "";
  duration.value = 0;
};
const ASRClick = async originalID => {
  const loading = ElLoading.service({
    lock: true,
    text: "Speech Recognition in progress..."
  });
  try {
    const ASRResult = await callASR({ videoID: originalID });
    textarea.value = ASRResult.data.result;
  } catch (error) {
    return { code: 400 };
  } finally {
    loading.close();
  }
  return { code: 200 };
};
const watchVideo = videoTime => {
  if (
    videoTime > modifiedTextList.value[modifiedTextList.value.length - 1].end
  ) {
    OperationMethod.hightLightInit();
    return OperationMethod.dragBackgroundColor(
      editAligned.value,
      modifiedTextList.value.length
    );
  }
  let watchSwitch = true;
  for (let index = 0; index < modifiedTextList.value.length; index++) {
    if (
      modifiedTextList.value[index].start < videoTime &&
      videoTime <= modifiedTextList.value[index].end
    ) {
      OperationMethod.highlightOver(
        index - 1,
        "rgba(181, 198, 241, 0.2)",
        "#fff"
      );
      OperationMethod.highlightOver(
        index,
        "rgba(255, 228, 196, 0.5)",
        "rgba(255, 228, 196, 0.5)"
      );
      OperationMethod.dragBackgroundColor(
        editAligned.value,
        modifiedTextList.value.length
      );
      watchSwitch = false;
      break;
    }
  }
  if (watchSwitch) {
    OperationMethod.hightLightInit();
    OperationMethod.dragBackgroundColor(
      editAligned.value,
      modifiedTextList.value.length
    );
  }
};
const modifiedVideoTimeEvent = that => {
  let count = 10;
  let timer = setInterval(() => {
    modifiedVideoTime.value = that.currentTime();
    modifiedWavesurfer.value.seekAndCenter(
      modifiedVideoTime.value / duration.value > 1
        ? 1
        : modifiedVideoTime.value / duration.value
    );
    watchVideo(modifiedVideoTime.value);
    count == 0 ? clearInterval(timer) : count--;
  }, 25);
};
const seekToTime = textitem => {
  if (textitem == null) {
    setTimeout(() => {
      modifiedPlayer.value.currentTime(
        modifiedWavesurfer.value.getCurrentTime() + 0.000001
      );
    }, 5);
  } else {
    modifiedPlayer.value.currentTime(textitem + 0.000001);
  }
};
const dragEnd = (e, type, method, lastkey) => {
  var element = document.elementFromPoint(e.clientX, e.clientY);
  if (
    $(element)
      .parent()
      .attr("class") == "textContent"
  ) {
    editAlignedIndex.value = Number($(element).attr("id"));
    editAligned.value = editAligned.value.filter(item => {
      return !(
        item[0] == editAlignedIndex.value &&
        item[1] == type &&
        item[2] == method
      );
    });
    let item = [editAlignedIndex.value, type, method, lastkey];
    if (method == "replace") {
      dialogReplaceText.value = true;
      dialogReplaceTitle.value =
        modifiedTextList.value[editAlignedIndex.value].text;
      return;
    }
    editAligned.value.push(item);
    OperationMethod.dragBackgroundColor(
      editAligned.value,
      modifiedTextList.value.length
    );
  }
};
const dragOver = e => {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
};
const modifiedTimeLineZoom = e => {
  modifiedWavesurfer.value.zoom((800 / duration.value) * e);
};
const modifiedWaveAudeo = () => {
  modifiedPlayer.value = videojs(
    "modified-my-player",
    {},
    function onPlayerReady() {
      this.on("play", function(e) {
        modifiedVideoTimeEvent(this);
      });
      this.on("timeupdate", function(e) {
        modifiedVideoTimeEvent(this);
      });
      this.on("seeked", function() {
        OperationMethod.hightLightInit();
      });
    }
  );
};
watchEffect(
  () => {
    if (activeStep.value == 2) {
      modifiedWaveAudeo();
    }
  },
  {
    flush: "post"
  }
);
const sliderInput = e => {
  editAligned.value = OperationMethod.modifyCard(
    editAligned.value,
    noiseObj.value,
    modifiedTextList.value.length
  );
};
watchEffect(() => {
  OperationMethod.dragBackgroundColor(
    editAligned.value,
    modifiedTextList.value.length
  );
});

const mouseDown = e => {
  var state = true;
  e.path.forEach(item => {
    if ($(item).attr("id") == "popup") {
      state = false;
    }
  });
  if (state && modifiedTextList.value.length) {
    rightPopup.value = false;
    OperationMethod.hightLightInit();
    OperationMethod.dragBackgroundColor(
      editAligned.value,
      modifiedTextList.value.length
    );
  }
};
const dialogConfirm = () => {
  dialogReplaceText.value = false;
  rightPopup.value = false;
  let item = [editAlignedIndex.value, "t", "replace", alternateText.value];
  editAligned.value.push(item);
  OperationMethod.dragBackgroundColor(
    editAligned.value,
    modifiedTextList.value.length
  );
};
const textOver = async (e, index, background, textBackground) => {
  if (!rightPopup.value) {
    await OperationMethod.highlightOver(index, background, textBackground);
  }
  OperationMethod.dragBackgroundColor(
    editAligned.value,
    modifiedTextList.value.length
  );
  overTooltip.value = {
    replace: false,
    remove: false,
    blank: false,
    gblur: false,
    mute: false,
    noise_white: false
  };
  if (textBackground != "#fff") {
    editAligned.value.forEach(item => {
      if (item[0] == index) {
        overTooltip.value[
          item[2] == "noise-white" ? "noise_white" : item[2]
        ] = true;
      }
    });
    triangleTop.value = $(`#${index}`).offset().top + $(`#${index}`).height();
    tooltipTop.value = triangleTop.value + 8;
    
    triangleLeft.value =
      $(`#${index}`).offset().left + $(`#${index}`).width() / 2 - 5 - $(`.appContainer`).offset().left;
    tooltipLeft.value = triangleLeft.value - 40;
    setTimeout(() => {
      tooltipLeft.value = triangleLeft.value - $(`.ztooltip`).width() / 2 + 2;
    }, 1);
  }
};
const waveOver = async (e, waveBackground, textBackground) => {
  if (
    e.target.getAttributeNode("class") &&
    e.target.getAttributeNode("class").value === "wavesurfer-region" &&
    !rightPopup.value
  ) {
    let index = $(`#modifiedWaveform > wave > region.wavesurfer-region`).index(
      e.target
    );
    await OperationMethod.highlightOver(index, waveBackground, textBackground);
  }
  OperationMethod.dragBackgroundColor(
    editAligned.value,
    modifiedTextList.value.length
  );
};
const modalityModify = (e, t, last = "") => {
  OperationMethod.hightLightInit();
  let editAligned = editAligned.value.filter(item => {
    return item[0] == editAligned.valueIndex && (item[1] == t || "all" == t)
      ? false
      : true;
  });
  if (e != "clear") {
    let lastkey = "";
    if (t == "v") {
      lastkey = e == "gblur" ? last : "";
    } else if (t == "a") {
      lastkey = e == "noise-white" ? "low" : "";
    } else if (e == "replace") {
      dialogReplaceText.value = true;
      dialogReplaceTitle.value =
        modifiedTextList.value[editAligned.valueIndex].text;
      return;
    }
    let item = [editAligned.valueIndex, t, e, lastkey];
    editAligned.push(item);
  } else {
    editAligned = [];
    noiseObj.value = {
      white: 0,
      metro: 0,
      office: 0,
      park: 0,
      diner: 0,
      traffic: 0
    };
    OperationMethod.dragBackgroundColor(
      editAligned,
      modifiedTextList.value.length
    );
  }
  editAligned.value = editAligned;
  rightPopup.value = false;
};
const rightClickItem = async (e, type) => {
  showPositionTop.value = e.pageY || 0;
  showPositionLeft.value = e.pageX || 0;
  rightPopup.value = true;
  await OperationMethod.hightLightInit();
  let index = $(
    type == "wave"
      ? "#modifiedWaveform > wave > region.wavesurfer-region"
      : ".pageContainer .modifiedCon .textContent span"
  ).index(e.target);
  editAligned.value.Index = index;
  await OperationMethod.highlightOver(
    index,
    "rgba(255, 228, 196, 0.5)",
    "rgba(255, 228, 196, 0.5)"
  );
  modalityDisplay.value.clear = false;
  editAligned.value.forEach(item => {
    if (item[0] == index) {
      modalityDisplay.value.clear = true;
    }
  });
  modalityDisplay.value.clear =
    noiseObj.value.white ||
    noiseObj.value.metro ||
    noiseObj.value.office ||
    noiseObj.value.park ||
    noiseObj.value.diner ||
    noiseObj.value.traffic ||
    modalityDisplay.value.clear
      ? true
      : false;
};
const generate = async () => {
  if (editAligned.value.length == 0) {
    ElMessage({
      message: "Please modify the video first.",
      type: "warning"
    });
    return;
  }
  let data = {
    videoID: originalID.value,
    words: editAligned.value
  };
  const loading = ElLoading.service({
    lock: true,
    text: "Generating modified video..."
  });
  try {
    var result = await videoEditAligned(data);
  } catch (error) {
    loading.close();
    return;
  }
  if (result.data.code == 200) {
    // McurrentVideoUrl.value = `${window.static_url}/${originalID.value}/modified_video.mp4`;
    clearModified();
    try {
      var response = await getFileFromUrl(
        `${originalID.value}/modified_video.mp4`,
        "modified_video.mp4"
      );
      McurrentVideoUrl.value = URL.createObjectURL(response)
    } catch (error) {
      loading.close();
      return;
    }
    addVideoModified([response]);
    modifiedPlayer.value.dispose();
    let html =
      '<video id="modified-my-player" class="video-js vjs-big-play-centered" controls="true"><source src="' +
      McurrentVideoUrl.value +
      '" type="video/mp4" /></video>';
    document.getElementById("modifiedContainer").innerHTML = html;
    modifiedWaveAudeo();
    modifiedWaveformSwitch.value = false;
    setTimeout(() => {
      modifiedWaveformSwitch.value = true;
      setTimeout(() => {
        modifiedEvent();
      }, 2);
    }, 2);
    editAligned.value.forEach(item => {
      if (item[1] == "t") {
        if (item[2] == "replace") {
          modifiedTextList.value[item[0]].text = item[3];
        } else {
          modifiedTextList.value[item[0]].text = "[REMOVED]";
        }
      }
    });
    loading.close();
  }
};
onMounted(() => {
  window.addEventListener("mousedown", mouseDown);
  arrayTitle.value = $(".elSteps").width() + 5;
});
</script>

<style scoped lang="scss">
.drag_box {
  position: absolute;
  font-size: 14px;
  line-height: 25px;
  z-index: 100;
  color: #616161;
  user-select: none;
}
.appContainer {
  margin:0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  width: 850px;
  overflow: hidden;
}
.stepTop {
  position: absolute;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  width: 100%;
  background-color: #fff !important;
  z-index: 99;
}
.elSteps {
  background-color: #fff;
}
.firstPageContainer {
  position: relative;
}
.firstPageContainer,
.pageContainer {
  padding-top: 80px;
  width: 800px;
  padding-bottom: 150px;
}
.modifiedCon {
  position: relative;
}
.tipNote {
  font-size: 14px;
  color: #5f5f5f;
  margin-top: 50px;
}
.upload-demo {
  margin-top: 15px;
  height: 260px;
}
.arrayTitle {
  position: relative;
  padding-top: 10px;
  font-size: 21px;
  color: #464646;
}
.firstPageContainer .elButton {
  width: 100px;
  height: 30px;
}
.firstPageContainer .aligned {
  margin-left: 150px;
}
.modifiedTopContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}
.topTip {
  font-size: 18px;
  color: #353535;
}
.topDescribe {
  font-size: 14px;
  color: #727272;
  margin-top: 5px;
}
.topDescribe span {
  font-weight: bold;
}
.methodContent {
  width: 320px;
  height: 340px;
  overflow: auto;
}
.methodContent::-webkit-scrollbar {
  width: 10px;
}

.methodContent::-webkit-scrollbar-track {
  background-color: rgb(255, 255, 255);
  box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
}

.methodContent::-webkit-scrollbar-thumb {
  background-color: rgb(207, 207, 207);
  border-radius: 10px;
}
.fixedIcon {
  position: absolute;
  right: 20px;
  height: 20px;
  width: 20px;
}
.video-js1 {
  margin-top: 15px;
  width: 800px;
  height: 500px;
}
.video-js {
  margin-top: 15px;
}
#app
  .firstPageContainer
  button.el-button.elButton.el-button--default.el-button--small {
  margin-left: 0;
}
.firstPageContainer .transcriptCon {
  width: 800px;
  display: flex;
  margin-top: 15px;
}
.firstPageContainer .transcriptTip {
  width: 100px;
}
.nextbutton {
  margin-top: 20px;
  position: absolute;
  width: 95px;
  right: 0;
}
.pageContainer .videoframe {
  overflow-x: scroll;
  margin-top: 20px;
  width: 800px;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  overflow-y: hidden;
  box-shadow: 0 0 1px 1px rgb(211, 211, 211);
  border-radius: 5px;
  height: 120px;
}
.modifiedWaveform {
  width: 800px;
  position: relative;
  margin-top: 20px;
  box-shadow: 0 0 1px 1px rgb(211, 211, 211);
  border-radius: 5px;
  overflow-y: hidden;
}
#waveform > wave::-webkit-scrollbar {
  height: 10px;
}
#waveform > wave::-webkit-scrollbar-track {
  background-color: rgb(255, 255, 255);
  box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
}
#waveform > wave::-webkit-scrollbar-thumb {
  background-color: rgb(207, 207, 207);
  border-radius: 10px;
}
.textContent {
  margin-top: 10px;
  padding-bottom: 20px;
  font-size: 18px;
  line-height: 25px;
}
.textContent span {
  display: inline-block;
  margin-top: 5px;
  margin-right: 10px;
  user-select: none;
}
.textContent span:hover {
  background-color: rgba(255, 228, 196, 0.5);
  cursor: pointer;
}
.popupContainer {
  box-shadow: 1px 1px 1px 1px rgb(189, 189, 189);
  width: 130px;
  background: #fff;
  padding: 10px;
  line-height: 30px;
  color: rgb(77, 77, 77);
  font-size: 16px;
  z-index: 99;
}
.popupContainer .modality {
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  line-height: 40px;
  align-items: center;
  font-size: 15px;
}
.popupContainer .modality span {
  display: inline-block;
  width: 70px;
  color: rgb(73, 73, 73);
}
.popupContainer .state {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  align-items: center;
  width: 100%;
  font-size: 14px;
  color: rgb(105, 105, 105);
}
.popupContainer .modalityClose {
  padding-left: 10px;
  font-size: 14px;
}
.popupContainer .modality:hover,
.popupContainer .modalityClose:hover,
.popupContainer .state:hover {
  cursor: pointer;
  background-color: rgb(236, 236, 236);
}
.generate {
  margin-left: 580px;
  margin-top: 20px;
  width: 95px;
}
.dialogReplaceTextContainer {
  display: flex;
  flex-direction: column;
  margin-top: -30px;
}
.dialogReplaceTitle {
  width: 100px;
}
.dialogReplaceItem {
  display: flex;
  flex-direction: row;
  align-items: center;
  margin-top: 10px;
}
.modalityMethod {
  display: flex;
  flex-direction: row;
  align-items: center;
  line-height: 25px;
  user-select: none;
}
.modalityMethod span {
  margin-top: -1.5px;
  margin-left: 10px;
  font-size: 15px;
}
.methodState {
  margin-left: 26px;
  font-size: 14px;
  position: relative;
  color: #616161;
  line-height: 25px;
  display: flex;
  flex-direction: row;
  align-items: center;
  user-select: none;
}
.methodStateNoise {
  margin-left: 26px;
  font-size: 14px;
  position: relative;
  color: #616161;
  line-height: 24px;
  display: flex;
  flex-direction: column;
  margin-top: 0.5px;
  user-select: none;
}
.noiseTitle {
  display: flex;
  flex-direction: row;
  align-items: center;
}
.noiseTitle span {
  margin-left: 5px;
  margin-top: -0.5px;
}
.noiseContainer {
  display: flex;
  flex-direction: column;
  line-height: 23px;
  margin-left: 20px;
  font-size: 13px;
  width: 235px;
}
.noiseItem {
  display: flex;
  flex-direction: row;
}
.noiseItem span {
  width: 100px;
  display: inline-block;
}
.methodStateNoise span:hover,
.modalityMethod span:hover,
.methodState:hover,
.dragDiv:hover {
  cursor: pointer;
}
.methodState span:hover {
  background-color: rgb(230, 230, 230);
}
.dragDiv {
  position: absolute;
  top: 200px;
  left: 100px;
  border: 1px solid black;
  user-select: none;
}
.ztooltip {
  position: absolute;
  padding: 10px;
  border-radius: 5px;
  background-color: rgb(29, 29, 29);
  color: aliceblue;
  display: flex;
  flex-direction: column;
  font-size: 14px;
  line-height: 22px;
}
.triangle {
  position: absolute;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 7px solid transparent;
  border-bottom: 10px solid rgb(29, 29, 29);
}
.selectMethod {
  position: relative;
  top: 10px;
}
.gblurSelect {
  margin-left: 20px;
  width: 90px;
}
.elSlider {
  width: 100px;
}
</style>

<style lang="scss">
.el-step.is-simple .el-step__arrow {
  flex-grow: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 50px !important;
  transform: scale(0.5);
}
.el-step.is-simple {
  display: flex;
  flex-direction: row;
  align-items: center;
  flex-basis: 0 !important;
}
.el-step.is-simple .el-step__icon {
  width: 16px;
  height: 20px;
  line-height: 20px;
}
.el-step.is-simple .el-step__main {
  position: relative;
  display: flex;
  align-items: stretch;
  flex-grow: 0;
  width: max-content;
}
.el-step.is-simple .el-step__title {
  font-size: 14px;
}
#app .stepTop .el-step__main .el-step__title {
  max-width: 100%;
}
.el-step:last-of-type.is-flex {
  width: 121px;
}
#modified-my-player {
  margin-top: 15px;
  width: 470px;
  height: 340px;
}
.stepTop .el-step__main .is-finish,
.stepTop .is-finish {
  color: #67c23a;
}
div.stepTop .is-process > div,
#app .stepTop .el-step__main .is-process {
  color: #409eff;
}
.el-upload-dragger .successUpload .el-icon--upload {
  color: #409eff;
}
.el-upload__tip {
  font-size: 13px;
  display: flex;
  flex-direction: row;
  align-items: center;
}
.el-upload__tip span {
  margin-left: 5px;
}
.el-card__body {
  padding: 10px;
}
.tooltip-base-box .box-item {
  width: 110px;
}
.el-dialog__body {
  height: 40px;
}
.el-dialog__footer {
  padding-top: 0;
}
#modifiedWaveform {
  height: 90px !important;
}
#modifiedWaveform > wave {
  height: 90px !important;
  &::-webkit-scrollbar {
    height: 10px;
  }
  &::-webkit-scrollbar-track {
    background-color: rgb(255, 255, 255);
    box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
  }
  &::-webkit-scrollbar-thumb {
    background-color: rgb(207, 207, 207);
    border-radius: 10px;
  }
}
.methodContent .el-slider__runway.show-input {
  margin-right: 10px;
}
.methodContent .el-slider__button {
  height: 14px;
  width: 14px;
}
.methodContent .el-slider__input {
  width: 60px;
}
.methodContent .el-input-number__decrease,
.methodContent .el-input-number__increase {
  width: 14px;
}
</style>