import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const stageNames={inpainting:'인페인트',detection:'감지',ocr:'OCR'};

app.registerExtension({
  name: 'Koharu.Workflows',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!['KoharuDetectionOutput','KoharuOCROutput'].includes(nodeData.name)) return;
    const previous = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(data) {
      previous?.call(this,data);
      if (!this.koharuResult) {
        const element=document.createElement('textarea');element.readOnly=true;
        element.style.cssText='width:100%;height:140px;resize:none;background:#171b22;color:#eee';
        this.addDOMWidget('koharu_result','text',element,{serialize:false});
        this.koharuResult=element;
      }
      this.koharuResult.value=data.text?.join('\n')??JSON.stringify(data.koharu??data,null,2);
    };
  },
  async setup() {
    const panel = document.createElement('section');
    panel.style.cssText = 'padding:16px;width:100%;box-sizing:border-box;overflow:auto;color:var(--input-text);font:13px sans-serif';
    const title = document.createElement('h3'); title.textContent = 'Koharu 워크플로우';
    panel.append(title);
    const body = document.createElement('div'); body.style.cssText='display:grid;gap:8px;margin-top:12px';
    const stage = document.createElement('select');
    for (const [value,label] of [['inpainting','인페인트'],['detection','감지'],['ocr','OCR']]) {
      const option=document.createElement('option');option.value=value;option.textContent=label;stage.append(option);
    }
    const token=document.createElement('input'); token.type='password';token.placeholder='서버 API 토큰 (설정한 경우)';token.autocomplete='off';
    const message=document.createElement('p');message.setAttribute('role','status');message.style.whiteSpace='pre-wrap';
    message.textContent='단계별 그래프를 열고 편집한 뒤 적용하세요. 적용 전 캔버스 편집은 서버 요청에 반영되지 않습니다.';
    const buttons=[];
    let loadedStage=null;
    async function request(path,data) {
      const headers={};if(token.value)headers.Authorization='Bearer '+token.value;
      if(data!==undefined)headers['Content-Type']='application/json';
      const response=await api.fetchApi(path,{method:data===undefined?'GET':'POST',headers,body:data===undefined?undefined:JSON.stringify(data)});
      const text=await response.text();if(!response.ok)throw new Error(text);
      return JSON.parse(text);
    }
    async function run(action) {
      buttons.forEach(b=>b.disabled=true);stage.disabled=true;
      try { await action(); } catch(error) {message.textContent=String(error.message??error);}
      finally {buttons.forEach(b=>b.disabled=false);stage.disabled=false;}
    }
    function button(label, action) {
      const b=document.createElement('button');b.textContent=label;b.onclick=()=>run(action);buttons.push(b);body.append(b);return b;
    }
    body.append(stage,token);
    const live = document.createElement('input'); live.type='checkbox';
    const liveLabel=document.createElement('label');
    liveLabel.append(live,document.createTextNode(' Koharu 요청 실시간 따라보기'));
    body.append(liveLabel);
    const watched = {};
    const choices=document.createElement('div');choices.style.cssText='display:flex;gap:12px;flex-wrap:wrap';
    for(const [name,label] of [['inpainting','인페인트'],['detection','감지'],['ocr','OCR']]){
      const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=name==='inpainting';
      watched[name]=checkbox;
      const item=document.createElement('label');item.append(checkbox,document.createTextNode(' '+label));choices.append(item);
      checkbox.onchange=()=>{void heartbeat();};
    }
    body.append(choices);
    let liveGraphLoaded=false, displayedSignature=null;
    const apiTemplates=new Map();
    async function heartbeat() {
      if(!live.checked||!api.clientId)return;
      try {await request('/koharu/watch',{client_id:api.clientId,enabled:true,stages:Object.keys(watched).filter(name=>watched[name].checked)});}
      catch(error){message.textContent=String(error.message??error);}
    }
    live.onchange=async()=>{
      if(live.checked){
        if(!confirm('Koharu 요청을 따라보면 저장된 워크플로우로 캔버스가 전환될 수 있습니다. 편집 중인 작업을 저장했나요?')){live.checked=false;return;}
        await heartbeat();
        message.textContent='실시간 따라보기 켜짐. 실행 노드·진행률·미리보기 결과를 이 브라우저에서 받습니다.';
      }else if(api.clientId){await request('/koharu/watch',{client_id:api.clientId,enabled:false});}
    };
    setInterval(()=>{void heartbeat();},30000);
    api.addEventListener('status',()=>{void heartbeat();});
    api.addEventListener('koharu_follow',async event=>{
      if(!live.checked)return;
      const data=event.detail;
      if(!watched[data.stage]?.checked){
        await request('/koharu/watch/ready',{client_id:api.clientId,request_id:data.request_id});return;
      }
      try{
        const apiGraph=structuredClone(data.prompt);
        for(const node of Object.values(apiGraph)){
          if(node.class_type.startsWith('Koharu')&&'request_id' in node.inputs)node.inputs.request_id='';
        }
        const signature=JSON.stringify([data.stage,data.workflow??apiGraph]);
        if(signature!==displayedSignature){
          if(data.workflow){
            if(!liveGraphLoaded)await app.loadGraphData(data.workflow);
            else { (app.rootGraph??app.graph).configure(structuredClone(data.workflow));app.canvas.setDirty(true,true); }
          }else if(apiTemplates.has(signature)){
            (app.rootGraph??app.graph).configure(structuredClone(apiTemplates.get(signature)));app.canvas.setDirty(true,true);
          }else{
            await app.loadApiJson(apiGraph,'Koharu '+stageNames[data.stage]);
            if(apiTemplates.size>=8)apiTemplates.delete(apiTemplates.keys().next().value);
            apiTemplates.set(signature,(app.rootGraph??app.graph).serialize());
          }
          liveGraphLoaded=true;displayedSignature=signature;
        }
        loadedStage=data.stage;stage.value=data.stage;
        await request('/koharu/watch/ready',{client_id:api.clientId,request_id:data.request_id});
        message.textContent='Koharu '+stageNames[data.stage]+' 실행 요청을 실시간으로 따라보는 중…';
      }catch(error){message.textContent=String(error.message??error);}
    });
    button('현재 적용 워크플로우 열기',async()=>{
      const name=stage.value;
      const data=await request('/koharu/workflows/'+name);
      if(data.workflow)await app.loadGraphData(data.workflow);
      else await app.loadApiJson(data.prompt,'Koharu '+stageNames[name]);
      liveGraphLoaded=false;displayedSignature=null;
      loadedStage=name;
      message.textContent='편집 중: '+stageNames[name]+'\n모델·프롬프트·옵션과 중간 노드를 편집한 뒤 Koharu에 적용하세요.';
    });
    button('새 기본 템플릿 열기',async()=>{
      const name=stage.value;
      const data=await request('/koharu/workflows/'+name+'?template=1');
      if(data.workflow)await app.loadGraphData(data.workflow);
      else await app.loadApiJson(data.prompt,'Koharu '+stageNames[name]+' 기본 템플릿');
      liveGraphLoaded=false;displayedSignature=null;loadedStage=name;
      message.textContent='새 템플릿을 열었습니다. 기존 서버 프로필은 유지됩니다. 옵션을 편집하고 적용하세요.';
    });
    button('현재 그래프를 Koharu에 적용',async()=>{
      const name=stage.value;
      if(loadedStage!==null&&loadedStage!==name)throw new Error('열어 둔 단계와 선택한 단계가 다릅니다. 적용할 단계를 선택하세요.');
      const data=await app.graphToPrompt();
      await request('/koharu/workflows/'+name,{prompt:data.output,workflow:data.workflow});
      loadedStage=name;
      message.textContent='서버에 저장됨. 다음 요청부터 반영됩니다. 모델을 변경했다면 아래 모델 준비 버튼을 누르세요.';
    });
    button('선택 단계 모델 준비',async()=>{
      message.textContent='모델 준비 중… ComfyUI 큐에서 실행됩니다.';
      const data=await request('/koharu/warmup',{stages:[stage.value]});
      message.textContent=data.ready?'선택 단계 준비 완료':'모델 준비 상태를 확인하세요.';
    });
    button('감지·OCR·인페인트 모두 준비',async()=>{
      message.textContent='전체 모델 준비 중… ComfyUI 큐에서 실행됩니다.';
      await request('/koharu/warmup',{});
      message.textContent='전체 준비 완료. Koharu에서 서버 주소를 저장·연결 확인하세요.';
    });
    body.append(message);panel.append(body);
    app.extensionManager.registerSidebarTab({
      id: 'koharu-workflows', icon: 'pi pi-sliders-h', title: 'Koharu',
      tooltip: 'Koharu 워크플로우 관리', type: 'custom',
      render: (element) => { element.replaceChildren(panel); }
    });
  }
});
