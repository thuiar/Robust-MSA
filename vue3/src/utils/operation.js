import config from "@/config";
import WaveSurfer from "wavesurfer.js";
import Timeline from "wavesurfer.js/dist/plugin/wavesurfer.timeline.js";
import Regions from "wavesurfer.js/dist/plugin/wavesurfer.regions.js";
import CursorPlugin from "wavesurfer.js/dist/plugin/wavesurfer.cursor.js";
const hightLightInit = () => {
  $(`#modifiedWaveform > wave > region.wavesurfer-region`).css({
    background: "rgba(181, 198, 241, 0.2)",
  });
  $(`.modifiedCon .textContent span`).css({
    background: "#fff",
  });
};

const highlightOver = (index, background, textBackground) => {
  $(
    `#modifiedWaveform > wave > region.wavesurfer-region:nth-child(${
      index + 3
    })`
  ).css({
    background,
  });
  $(
    `.pageContainer .modifiedCon .textContent span:nth-child(${index + 1})`
  ).css({
    background: textBackground,
  });
};

const dragBackgroundColor = (editAligned, textLength) => {
  var backgroundColor = [];
  var state = false;
  editAligned.forEach((item) => {
    if (item[0] == -1) {
      state = true;
    }
  });
  if (state) {
    for (let i = 0; i < textLength; i++) {
      backgroundColor.push([i, 1]);
    }
  }
  editAligned.forEach((item) => {
    let index = -1;
    backgroundColor.forEach((color) => {
      if (item[0] == color[0]) {
        if (color[1] == 0) {
          if (item[1] == "t") {
            index = 4;
          } else if (item[1] == "a") {
            index = 3;
          } else {
            index = 7;
          }
        } else if (color[1] == 1) {
          if (item[1] == "t") {
            index = 5;
          } else if (item[1] == "v") {
            index = 3;
          } else {
            index = 7;
          }
        } else if (color[1] == 2) {
          if (item[1] == "v") {
            index = 4;
          } else if (item[1] == "a") {
            index = 5;
          } else {
            index = 7;
          }
        } else if (color[1] == 3) {
          if (item[1] == "t") {
            index = 6;
          } else {
            index = 7;
          }
        } else if (color[1] == 4) {
          if (item[1] == "a") {
            index = 6;
          } else {
            index = 7;
          }
        } else if (color[1] == 5) {
          if (item[1] == "v") {
            index = 6;
          } else {
            index = 7;
          }
        }
      }
    });
    if (index == -1) {
      if (item[1] == "v") {
        index = 0;
      } else if (item[1] == "a") {
        index = 1;
      } else if (item[1] == "t") {
        index = 2;
      }
    } else {
      if (index != 7) {
        backgroundColor = backgroundColor.filter((itemColor) => {
          return itemColor[0] != item[0];
        });
      }
    }
    if (index != 7) {
      backgroundColor.push([item[0], index]);
    }
  });
  backgroundColor.forEach((item) => {
    $(`#${item[0]}`).css({
      background: config.legend[item[1]]["bg"],
    });
  });
};
const _timeInterval = (pxPerSec) => {
  var retval = 1;
  if (pxPerSec >= 3000) {
    retval = 0.005;
  } else if (pxPerSec >= 1000) {
    retval = 0.01;
  } else if (pxPerSec >= 500) {
    retval = 0.05;
  } else if (pxPerSec >= 200) {
    retval = 0.1;
  } else if (pxPerSec >= 100) {
    retval = 0.4;
  } else if (pxPerSec >= 80) {
    retval = 1;
  } else if (pxPerSec >= 60) {
    retval = 2;
  } else if (pxPerSec >= 40) {
    retval = 1;
  } else if (pxPerSec >= 20) {
    retval = 5;
  } else {
    retval = Math.ceil(0.5 / pxPerSec) * 60;
  }
  return retval;
};
const echartsDataChange = (methodDetailValue) => {
  if (methodDetailValue == "F1amplitude") {
    return "F1amplitude: Ratio of the energy of the spectral harmonic peak at the first formant's centre frequency to the energy of the spectral peak at F0.";
  } else if (methodDetailValue == "F1bandwidth") {
    return "F1bandwidth: Bandwidth of first formant.";
  } else if (methodDetailValue == "F1frequency") {
    return "F1frequency: Centre frequency of first formant.";
  } else if (methodDetailValue == "HNR") {
    return "HNRdBACF: Harmonics-to-Noise Ratio. Relation of energy in harmonic components to energy in noise-like components.";
  } else if (methodDetailValue == "alphaRatio") {
    return "alphaRatio: Ratio of the summed energy from 50-1000 Hz and 1-5 kHz.";
  } else if (methodDetailValue == "loudness") {
    return "Loudness: Estimate of perceived signal intensity from an auditory spectrum.";
  } else if (methodDetailValue == "mfcc1") {
    return "MFCC1: Mel frequency cepstral coefficients - 1 ";
  } else if (methodDetailValue == "mfcc2") {
    return "MFCC2: Mel frequency cepstral coefficients - 2.";
  } else if (methodDetailValue == "mfcc3") {
    return "MFCC3: Mel frequency cepstral coefficients - 3.";
  } else if (methodDetailValue == "mfcc4") {
    return "MFCC4: Mel frequency cepstral coefficients - 4 ";
  } else if (methodDetailValue == "pitch") {
    return "Pitch: logarithmic F0 on a semitone frequency scale, starting at 27.5 Hz (semitone 0).";
  } else if (methodDetailValue == "AU01_r") {
    return "Action Unit 01: Inner brow raiser.";
  } else if (methodDetailValue == "AU02_r") {
    return "Action Unit 02: Outer brow raiser.";
  } else if (methodDetailValue == "AU04_r") {
    return "Action Unit 04: Brow lowerer.";
  } else if (methodDetailValue == "AU05_r") {
    return "Action Unit 05: Upper lid raiser.";
  } else if (methodDetailValue == "AU06_r") {
    return "Action Unit 06: Cheek raiser.";
  } else if (methodDetailValue == "AU07_r") {
    return "Action Unit 07: Lid rightener.";
  } else if (methodDetailValue == "AU09_r") {
    return "Action Unit 09: Nose wrinkler.";
  } else if (methodDetailValue == "AU10_r") {
    return "Action Unit 10: Upper lip raiser.";
  } else if (methodDetailValue == "AU12_r") {
    return "Action Unit 12: Lip corner puller.";
  } else if (methodDetailValue == "AU14_r") {
    return "Action Unit 14: Dimpler.";
  } else if (methodDetailValue == "AU15_r") {
    return "Action Unit 15: Lip corner depressor.";
  } else if (methodDetailValue == "AU17_r") {
    return "Action Unit 17: Chin raiser";
  } else if (methodDetailValue == "AU20_r") {
    return "Action Unit 20: Lip stretcher.";
  } else if (methodDetailValue == "AU23_r") {
    return "Action Unit 23: Lip tightener.";
  } else if (methodDetailValue == "AU25_r") {
    return "Action Unit 25: Lips part.";
  } else if (methodDetailValue == "AU26_r") {
    return "Action Unit 26: Jaw drop.";
  } else if (methodDetailValue == "AU45_r") {
    return "Action Unit 46: Wink.";
  } else {
    return "";
  }
};
const modifyCard = (editAligned, noiseObj, length) => {
  if (noiseObj.white) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.white &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_white"
      );
    });
    editAligned.push([-1, "a", "noise_white", noiseObj.white]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.white &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_white"
      );
    });
  }
  if (noiseObj.metro) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.metro &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_metro"
      );
    });
    editAligned.push([-1, "a", "noise_metro", noiseObj.metro]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.metro &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_metro"
      );
    });
  }
  if (noiseObj.office) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.office &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_office"
      );
    });
    editAligned.push([-1, "a", "noise_office", noiseObj.office]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.office &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_office"
      );
    });
  }
  if (noiseObj.park) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_park"
      );
    });
    editAligned.push([-1, "a", "noise_park", noiseObj.park]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_park"
      );
    });
  }
  if (noiseObj.diner) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_diner"
      );
    });
    editAligned.push([-1, "a", "noise_diner", noiseObj.diner]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_diner"
      );
    });
  }
  if (noiseObj.traffic) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.traffic &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_traffic"
      );
    });
    editAligned.push([-1, "a", "noise_traffic", noiseObj.traffic]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.traffic &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_traffic"
      );
    });
  }
  hightLightInit();
  dragBackgroundColor(editAligned, length);
  return editAligned;
};

const waveformCreate = (
  modifiedWaveform,
  regions,
  container = "#modifiedWaveTimeline"
) => {
  return WaveSurfer.create({
    container: modifiedWaveform,
    scrollParent: true,
    hideScrollbar: false,
    waveColor: "#409EFF",
    progressColor: "blue",
    backend: "MediaElement",
    mediaControls: false,
    audioRate: "1",
    plugins: [
      Timeline.create({
        container: container,
        timeInterval: _timeInterval,
      }),
      Regions.create({
        showTime: true,
        regions: regions,
      }),
      CursorPlugin.create({
        showTime: true,
        opacity: 1,
        customShowTimeStyle: {
          "background-color": "#000",
          color: "#fff",
          padding: "5px",
          "font-size": "10px",
        },
      }),
    ],
  });
};
const getVideoDuration = videoFile =>
  new Promise((resolve, reject) => {
    try {
      const url = URL.createObjectURL(videoFile);
      const tempAudio = new Audio(url);
      tempAudio.addEventListener("loadedmetadata", () => {
        resolve(tempAudio.duration * 1000000);
      });
    } catch (error) {
      console.log("getVideoDuration error", error);
      throw error;
    }
  });
  
export default {
  hightLightInit,
  highlightOver,
  dragBackgroundColor,
  modifyCard,
  waveformCreate,
  echartsDataChange,
  getVideoDuration
};
