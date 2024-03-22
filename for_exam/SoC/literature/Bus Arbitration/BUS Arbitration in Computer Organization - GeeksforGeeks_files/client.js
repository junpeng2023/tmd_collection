// v0.0.21 — Sun Jul 23 2023 12:38:40 GMT+0300 (Eastern European Summer Time)
var A=Object.defineProperty,H=Object.defineProperties;var j=Object.getOwnPropertyDescriptors;var S=Object.getOwnPropertySymbols;var U=Object.prototype.hasOwnProperty,q=Object.prototype.propertyIsEnumerable;var v=(r,e,t)=>e in r?A(r,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):r[e]=t,m=(r,e)=>{for(var t in e||={})U.call(e,t)&&v(r,t,e[t]);if(S)for(var t of S(e))q.call(e,t)&&v(r,t,e[t]);return r},l=(r,e)=>H(r,j(e));var h=(r,e,t)=>new Promise((o,i)=>{var n=c=>{try{a(t.next(c))}catch(g){i(g)}},d=c=>{try{a(t.throw(c))}catch(g){i(g)}},a=c=>c.done?o(c.value):Promise.resolve(c.value).then(n,d);a((t=t.apply(r,e)).next())});var W=/bidder_debug=true/.test(window.location.href),$=["%cBidder Debug %cv0.0.21","color: #fff; background-color: #02224C; padding: 5px","color: #fff; background-color: #0054c2; padding: 5px"],f={debug(...r){W&&console.log(...$,...r)},error(...r){console.error(...$,...r),fetch("https://rt.marphezis.com/client/logger",{method:"POST",body:["v0.0.21",...r].map(e=>e&&(e instanceof Object?JSON.stringify(e):e.trim())).join(" ")})},renderAdError(r,e){fetch("https://rt.marphezis.com/render/error/logger",{method:"POST",body:`<!-- v0.0.21  dp: ${e} --> 
 ${btoa(r)}`})}};function s(r,e){return f.error("Error: ",r,e),{code:r,message:e}}function u(r){return new Promise(e=>setTimeout(()=>{e()},r*1e3))}function P(r,e,t,o){return f.debug("Normalize bids"),r.map((i,n)=>l(m({},i),{index:n,size:e,settings:t,isLastBid:r.length-1===n,loop:o}))}function G(r){let e=Math.floor(Math.random()*Date.now()),t=r.includes("?")?"&":"?";return`${r}${t}_cb=${e}`}function V(r){return h(this,null,function*(){f.debug("Fetching Refresh Bidder Data");let e=yield fetch(G("https://rt.marphezis.com/refresh"),{method:"POST",body:JSON.stringify(r),credentials:"include"});if(e.status===200)return e.json();if(e.status!==204)throw Error(`Status: ${e.status}`)})}function _(r,e=1){return h(this,null,function*(){let{refreshBody:t,timeout:o,onError:i,onEmptyResponse:n}=r,d;try{f.debug(`Refresh Bidder Data after ${o*e} seconds`),yield u(o*e),d=yield V(l(m({},t),{timeoutMultiply:e}))}catch(a){s("fetch_refresh_error",a.message),i&&i()}return d||n(),d||_(r,e+1)})}function T(r,e){return[r,e]}function w(r,e=[]){let t=e.reduce((o,i)=>{let[n,d]=i;return o[n]=o[n]||[],o[n].push(d),o},{});return{send(o,i){if(r[o]){let n=new Image;n.src=`${r[o]}&${new URLSearchParams(l(m({},i),{ver:"0.0.21"})).toString()}`}t[o]&&t[o].forEach(n=>n())},get(o){return t[o]&&t[o].forEach(i=>i()),r[o]?r[o]+"&ver=0.0.21":void 0}}}var z=(r,{ad:e,dpid:t,size:o,settings:i},n)=>{let d={ad:e,bidder:t,size:`${o.width}x${o.height}`};try{i.confiantKey&&window.confiant&&window.confiant.services&&window.confiant.services().wrap!==void 0?window.confiant.services().wrap(r,d,i.confiantKey,(a,c,g,x,y,I,B)=>{console.log("Ad was blocked by confiant:",JSON.stringify({blockingType:a,blockingId:c,isBlocked:g,propertyId:x,tagId:y,impressionData:I,wasRendered:B},null,4))}):(s("confiant_obj_not_found"),n())}catch(a){s("confiant_render_error",a.message),n()}};function K(r,e){let t=r;try{t=decodeURIComponent(r)}catch(o){s("decode_ad",`${e.dpid} \u2014 ${btoa(r)}`)}return t}function Q(r){var i;let o=new DOMParser().parseFromString(r,"text/html").getElementsByTagName("script");try{for(let n of o)if(!n.type||n.type==="text/javascript"){let d=(i=n==null?void 0:n.textContent)==null?void 0:i.trim();new Function(d||"")}}catch(n){return!1}return!0}function X(r,e,t){try{return`<html lang="en">
				<body style="margin: 0; padding: 0; overflow: hidden;">
					<script>
						function sendPixel(pixel) { if (pixel) {const pixelImage = new Image(); pixelImage.src = pixel;}}
                        sendPixel("${e}")
					<\/script>
					${K(r.ad,r)}
					${t?`<script> sendPixel("${t}") <\/script>`:""}
				</body>
			</html>
			`}catch(o){throw s("get_ad_template",`${o.message}, ${r.dpid} - "${btoa(r.ad)}"`)}}function M(r,e,t){var d;if(!e.ad)throw s("ad_is_empty",`${e.dpid}`);if(!window.document.body)throw s("body_element_is_not_ready");window.document.body.appendChild(r);let o=(d=r.contentWindow)==null?void 0:d.document;if(!o)throw s("iframe_document_not_found");let i=t.get("impression"),n=t.get("rendered");if(o&&i){let a=X(e,i,n);if(!Q(a))throw f.renderAdError(e.ad,e.dpid),s("render_ad_syntax_error");t.send("nurl"),t.send("served");try{e.settings.confiantEnabled?(t.send("confiant"),z(o,l(m({},e),{ad:a}),()=>{o.write(a)})):o.write(a)}catch(c){throw s("render_ad",c.message)}}}function D({width:r,height:e}){try{let t=document.createElement("iframe");return t.width=`${r}`,t.height=`${e}`,t.style.border="none",t}catch(t){throw s("iframe_not_created",t.message)}}var E;(n=>{let r=!1,e;n.eids=[];function o(){window.IntentIqObject?r||(f.debug("Initialise IIQ"),r=!0,e=new window.IntentIqObject({partner:182805994,callback:d=>{n.eids=d.eids||[]},timeoutInMillis:1e3,sourceMetaData:"",shouldClearDuplicatresForRubicon:!1,manualWinReportEnabled:!0,reportDelay:0,ABTestingConfigurationSource:"percentage",abPercentage:95})):s("no_iiq_object")}n.fetchEids=o;function i(d,a){window.IntentIqObject&&e&&e.reportExternalWin({biddingPlatformId:4,cpm:d,bidderCode:a,currency:"USD"})}n.report=i})(E||(E={}));var R;(o=>{let r=!1;function t(){window.ID5?r||(window.ID5.init({partnerId:1105}).onAvailable(i=>{o.id=i.getUserId()}),r=!0):s("no_id5_object")}o.fetchId5=t})(R||(R={}));function C(r,e=0,t){return h(this,null,function*(){let{settings:o,pixels:i,size:n,refreshBody:d}=r,{timeout:a}=o,{send:c}=w(i),g=P(r.bids,n,o,e);e===0&&c("onstart"),t==null||t.remove(),o.id5Enabled&&R.fetchId5();let x=Date.now();for(let p of g){let N=w(p.pixels,[T("served",()=>E.report(p.cpm,p.dpid))]),{isLastBid:F,index:L,loop:k}=p;try{f.debug(`Render ad ${L+1} loop ${k}`),t=D(n),M(t,p,N),F||(yield u(a),t==null||t.remove())}catch(O){t==null||t.remove(),N.send("error",{err:O.code,message:O.message})}}o.iiqEnabled&&E.fetchEids();let y=x+30*1e3-a*1e3,I=Date.now();if(I<y){let p=Math.round((y-I)/1e3);f.debug(`Wait for ${p} seconds to reach minimum ${30} seconds timeout before refresh request`),yield u(p)}let B=yield _({refreshBody:l(m({},d),{eids:E.eids,id5:R.id}),timeout:a,onError:()=>c("refreshError",{err:"fetch_refresh_error"}),onEmptyResponse:()=>{f.debug("Missed opportunity"),c("missed")}});return C(B,e+1,t)})}window.onload=()=>{let r=window.__bidder_data;C(r)};