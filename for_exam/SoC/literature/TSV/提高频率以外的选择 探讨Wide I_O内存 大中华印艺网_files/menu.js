function isMatch(str1,str2){
	var index = str1.indexOf(str2);
	if(index==-1) return false;
	return true;
}

function ResumeError(){
	return true;
}
window.onerror = ResumeError;

function doClick(o){
	o.className="nav_current";
	var j;
	var id;
	var e;
	for(var i=1;i<=9;i++){
		id ="nav"+i;
		j = document.getElementById(id);
		e = document.getElementById("sub"+i);
		if(id != o.id){
			j.className="nav_link";
			e.style.display = "none";
		}else{
			e.style.display = "block";
		}
	}
}

function addfa(){                                                                                 
window.external.AddFavorite(sitepath, sitename);                                                                                 
}