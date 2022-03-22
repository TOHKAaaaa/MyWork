<template>
  <div>
    <el-row :gutter="100">
      <el-col :span="200">
        <el-card shadow="hover" class="box-card">
          <p >Question:</p><p>{{ getQuestionTitle()}}</p>
          <el-divider style="margin-top: 200px"><i class="el-icon-edit"></i></el-divider>
          <div>
            <el-row>
              <el-col :span="1"><p>A.</p></el-col>
              <el-col :span="2">
                <el-button :type="buttonState[0]? 'success': 'danger'" @click="sendAns(0)">{{ buttonText[0] }}</el-button>
              </el-col>
            </el-row>
            <el-row>
              <el-col :span="1"><p>B.</p></el-col>
              <el-col :span="2">
                <el-button :type="buttonState[1]? 'success': 'danger'" @click="sendAns(1)">{{ buttonText[1] }}</el-button>
              </el-col>
            </el-row>
            <el-row>
              <el-col :span="1"><p>C.</p></el-col>
              <el-col :span="2">
                <el-button :type="buttonState[2]? 'success': 'danger'" @click="sendAns(2)">{{ buttonText[2] }}</el-button>
              </el-col>
            </el-row>
            <el-row>
              <el-col :span="1"><p>D.</p></el-col>
              <el-col :span="1">
                <el-button :type="buttonState[3]? 'success': 'danger'" @click="sendAns(3)">{{ buttonText[3] }}</el-button>
              </el-col>
            </el-row>
          </div>
        </el-card>
      </el-col>
      <el-col :span="200">
        <el-card shadow="hover" class="box-card2">
          <ve-radar class="card2" :extend="chartExtend" :data="chartData"></ve-radar>
        </el-card>
      </el-col>
    </el-row>



  </div>
</template>

<script>
import { getAPI } from "@/axios";

const questionindex = [7, 8, 15, 92, 59, 50]

const index2Name = {
  7: 'Mode',
  8: 'Mean',
  15: 'Fraction of',
  92: 'Rotations',
  59: 'Exponents',
  50: 'Pythagorean Theorem'
}
export default {
  name: "Prediction",

  data() {
    return {
      buttonText: [],
      buttonState: [],
      quesIndex: 7,
      result: true,
      APIresult: [],
      chartExtend: {
        legend: {
          // align:'auto',
          // top:'top'
          type: 'scroll',
          bottom: 10,
          // y:'bottom'
        },

        grid: {
          top: 'top',
          height:"700px",
          width:"600px"
        },
        radar: {
          // shape: 'circle',
          name: {
            textStyle: {
              color: '#000',
              // backgroundColor: '#999',
              borderRadius: 3,
              padding: [3, 5]
            }
          },
          radius:120,
          center: ['50%', '40%'],
          indicator: [

            {
              name: index2Name[7],
              max: 1
            },
            {
              name: index2Name[8],
              max: 1
            },
            {
              name: index2Name[15],
              max: 1
            },
            {
              name: index2Name[92],
              max: 1
            },
            {
              name: index2Name[59],
              max: 1
            },
            {
              name: index2Name[50],
              max: 1
            },
          ]
        },
      },

      chartData: {
        dataType: 'percent',
        columns: ['time', index2Name[7], index2Name[8], index2Name[15], index2Name[92], index2Name[59], index2Name[50]],
        rows: []
      }
    }
  },
  created() {
    this.changeButtionState()
  },
  methods: {
    changeButtionState() {
      this.buttonText = ['错误', '错误', '错误']
      this.buttonState = [false, false, false]
      let place = Math.floor(Math.random() * 4)
      this.buttonState.splice(place, 0, true)
      this.buttonText.splice(place, 0, '正确')
    },
    nextQuestion() {
      return questionindex[Math.floor(Math.random() * questionindex.length)]
    },
    getQuestionTitle() {
      return this.quesIndex + '-' + index2Name[this.quesIndex]
    },
    sendAns(index) {
      this.result = this.buttonState[index]
      this.predict()
      this.quesIndex = this.nextQuestion()
      this.changeButtionState()
    },
    // sendWrong() {
    //   this.result = false
    //   this.predict()
    //   this.quesIndex = this.nextQuestion()
    //   this.changeButtionState()
    // },
    // sendRight() {
    //   this.result = true
    //   this.predict()
    //   this.quesIndex = this.nextQuestion()
    //   this.changeButtionState()
    // },
    async predict() {

      const {data: res} = await getAPI.get('/predict', {
        params: {
          quesIndex: this.quesIndex,
          result: this.result
        }
      })
      console.log(res.data)
      this.chartData.rows = res.data

    }
  }
}
</script>

<style>
.box-card {
  width: 480px;
  height:480px;
  margin-left: 100px;
  margin-top: 90px;
}
.box-card2 {
  width: 700px;
  height:600px;
  display: flex;
  flex-direction: column;
  justify-content:center;
}
.card2{
  width:700px;
  height:600px;
}
.el-row {
  margin-top: 40px;
  margin-left: 10px;
}
.el-col {
  margin-left: 20px;
}
.p {
  font-size: 200%;
  margin-top: 5px;
  font-width: bold;
}
body {
  background-color: #CDDFF1;
}
</style>

