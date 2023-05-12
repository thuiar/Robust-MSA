<template>
  <div>
    <div class="selectMethod">
      <div class="methodTitle">Methods</div>
      <div class="description">Please select at least one model!</div>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">Data Level Defence:</div>
      <el-checkbox-group class="checkboxGroup" v-model="dataDefense" @change="selectMethod">
        <el-checkbox label="Audio Denoising" />
        <el-checkbox label="Video MCI (time-consuming)" />
      </el-checkbox-group>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">Feature Level Defence:</div>
      <el-checkbox-group class="checkboxGroup" v-model="featureDefense" @change="selectMethod">
        <el-checkbox label="Feature Interpolation" />
      </el-checkbox-group>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">MSA Models:</div>
      <el-checkbox-group class="checkboxGroup" v-model="msaModel" @change="selectMethod">
        <el-checkbox label="TFN" />
        <el-checkbox label="LMF" />
        <el-checkbox label="MAG-BERT" />
        <el-checkbox label="MISA" />
        <!-- <el-checkbox label="MulT" /> -->
        <el-checkbox label="MMIM" />
        <el-checkbox label="Self-MM" />
        <el-checkbox label="TFR-Net" />
        <el-checkbox label="NIAT" />
      </el-checkbox-group>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
const dataDefense = ref([]);
const featureDefense = ref([]);
const msaModel = ref(["TFN","LMF","MAG-BERT","MISA","MMIM","Self-MM","TFR-Net",'NIAT']);
const emit = defineEmits(["transmitMethods"]);
const selectMethod = () => {
  let msaList = [];
  msaModel.value.forEach(item => {
    if (item == "LMF") {
      msaList.push("lmf");
    } else if (item == "MAG-BERT") {
      msaList.push("bert_mag");
    } else if (item == "MISA") {
      msaList.push("misa");
    } else if (item == "MulT") {
      msaList.push("mult");
    } else if (item == "Self-MM") {
      msaList.push("self_mm");
    } else if (item == "TFR-Net") {
      msaList.push("tfr_net");
    } else if (item == "TFN") {
      msaList.push("tfn");
    } else if (item == "MMIM") {
      msaList.push("mmim");
    }else if (item == "NIAT") {
      msaList.push("niat");
    }
  });
  let defenseList = [];
  featureDefense.value.forEach(item => {
    if (item == "Feature Interpolation") {
      defenseList.push("f_interpol");
    }
  });
  dataDefense.value.forEach(item => {
    if (item == "Audio Denoising") {
      defenseList.push("a_denoise");
    } else if (item == "Video MCI (time-consuming)") {
      defenseList.push("v_reconstruct");
    }
  });
  let methods = { defence: defenseList, models: msaList };
  emit("transmitMethods", methods);
};
</script>
<style scoped>
.selectContainer {
  display: flex;
  flex-direction: row;
  align-items: center;
}
.selectTitle {
  width: 100px;
}
.selectMethod {
  display: flex;
  flex-direction: column;
  padding-bottom: 20px;
}
.methodTitle {
  font-size: 18px;
  color: #2c2c2c;
  line-height: 35px;
}
.description {
  font-size: 15px;
  color: #7a7a7a;
  line-height: 20px;
}
.checkboxContainer {
  display: flex;
  flex-direction: row;
  padding-bottom: 10px;
}
.checkboxTitle {
  width: 195px;
  font-size: 16px;
  color: #2c2c2c;
  line-height: 30px;
}
.checkboxGroup {
  width: 600px;
}
</style>
<style>
.selectMethod .checkboxContainer .el-checkbox__label {
  font-size: 16px;
}
</style>