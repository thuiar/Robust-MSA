<template>
  <div class="legendContainer">
    <div class="legendTitle">Legend:</div>
    <div class="legendContent">
      <div class="item" v-for="item in legend" :key="item">
        <div class="color" :style="{
            backgroundColor:item.color
        }"></div>
        <div class="tip">{{item.tip}}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import config from "../config";
import { ref } from "vue";
const legend = ref(config.legend);
</script>

<style scoped>
.legendContainer {
  display: flex;
  flex-direction: row;
  width:800px;
  line-height: 30px;
  color: rgb(80, 80, 80);
  user-select: none;
}
.legendTitle{
  font-size: 16px;
}
.legendContent {
  margin-left: 10px;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
}
.color {
  width: 17px;
  height: 17px;
  border-radius: 5px;
}
.item {
  display: flex;
  flex-direction: row;
  align-items: center;
  font-size: 14px;
}
.tip {
  margin-left: 4px;
  width: 220px;
}
</style>