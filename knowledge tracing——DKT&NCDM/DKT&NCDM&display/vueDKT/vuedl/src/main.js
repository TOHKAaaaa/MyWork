import Vue from 'vue'
import App from './App.vue'
import 'bootstrap/dist/css/bootstrap.min.css'
import router from './routes'
import './plugins/element.js'
import VCharts from 'v-charts'

// import 'element-ui/lib/theme-chalk/index.css'
// import element from './element/index'
// Vue.use(element)
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
Vue.use(ElementUI);
// import echarts from 'echarts'
Vue.use(VCharts)
// Vue.prototype.$echarts = echarts;

Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App),
}).$mount('#app')
