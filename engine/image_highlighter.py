"""
Label QC Checker Pro - Coordinate-Accurate Image Highlighter
"""

from pathlib import Path
import re
import uuid
import cv2
import numpy as np
from config import OUTPUT_DIR


class ImageHighlighter:
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 180, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __init__(self):
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_text(self, value):
        if value is None:
            return ""
        s = str(value).upper()
        for a, b in {"—":"-","–":"-","−":"-","_":"-","|":"I"}.items():
            s = s.replace(a,b)
        return re.sub(r"\s+", " ", s).strip()

    def normalize_for_match(self, value):
        return re.sub(r"[^A-Z0-9]", "", self.normalize_text(value))

    def clean_word(self, value):
        if value is None:
            return []
        return re.findall(r"[A-Za-z0-9]+(?:[-./%][A-Za-z0-9]+)*", str(value))

    def _safe_box(self, image, item, padding=3):
        if not isinstance(item, dict):
            return None
        try:
            x1 = int(item.get("x", 0)); y1 = int(item.get("y", 0))
            w = int(item.get("w", 0)); h = int(item.get("h", 0))
            x2 = int(item.get("x2", x1+w)); y2 = int(item.get("y2", y1+h))
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        H, W = image.shape[:2]
        x1=max(0,min(x1-padding,W-1)); y1=max(0,min(y1-padding,H-1))
        x2=max(0,min(x2+padding,W-1)); y2=max(0,min(y2+padding,H-1))
        return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None

    def _draw_box(self, image, box, color, label=None, thickness=3):
        if not box:
            return
        x1,y1,x2,y2=box
        cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness,cv2.LINE_AA)
        if label:
            font=cv2.FONT_HERSHEY_SIMPLEX
            scale=0.42
            tw,th=cv2.getTextSize(str(label)[:24],font,scale,1)[0]
            ly=max(th+6,y1-3)
            cv2.rectangle(image,(x1,ly-th-6),(min(image.shape[1]-1,x1+tw+8),ly+3),color,-1)
            cv2.putText(image,str(label)[:24],(x1+4,ly),font,scale,self.WHITE,1,cv2.LINE_AA)

    def _line_index(self, item):
        return item.get("line_index", item.get("ocr_line_index", None))

    def _find_on_line(self, words, targets, line_index=None):
        targets=[self.normalize_for_match(x) for x in targets if self.normalize_for_match(x)]
        if not targets or not words:
            return []
        candidates=[]
        for idx,item in enumerate(words):
            if not isinstance(item,dict) or not item.get("text"):
                continue
            if line_index is not None and self._line_index(item) != line_index:
                continue
            norm=self.normalize_for_match(item.get("text"))
            if not norm:
                continue
            candidates.append((idx,item,norm))
        used=set(); found=[]
        # Exact/contained, left-to-right. Each target can only consume one word.
        for target in targets:
            best=None
            for idx,item,norm in candidates:
                if idx in used: continue
                score=2 if norm==target else (1 if len(target)>=3 and (target in norm or norm in target) else 0)
                if score and (best is None or score>best[0]):
                    best=(score,idx,item)
            if best:
                used.add(best[1]); found.append(best[2])
        return found

    def _find_phrase_match(self, words, phrase, line_index=None):
        targets=[self.normalize_for_match(x) for x in self.clean_word(phrase)]
        if not targets: return []
        line_words=[]
        for idx,item in enumerate(words or []):
            if not isinstance(item,dict): continue
            if line_index is not None and self._line_index(item)!=line_index: continue
            norm=self.normalize_for_match(item.get("text",""))
            if norm:
                line_words.append((idx,item,norm))
        line_words.sort(key=lambda z:(z[1].get("x",0),z[1].get("y",0)))
        n=len(targets)
        for start in range(max(0,len(line_words)-n+1)):
            cand=line_words[start:start+n]
            if [x[2] for x in cand]==targets:
                return [x[1] for x in cand]
        return self._find_on_line(words, targets, line_index)

    def _save(self, image, output_path):
        output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(output_path),image):
            raise IOError(f"Unable to save highlighted image: {output_path}")
        return output_path

    def create_output_path(self,prefix="highlighted",extension=".jpg"):
        return self.output_dir/f"{prefix}_{uuid.uuid4().hex}{extension if str(extension).startswith('.') else '.'+str(extension)}"

    def highlight_visual_differences(self, image, words, differences, side, output_path):
        """
        Draw only the words belonging to the requested side.
        line_index prevents a repeated word elsewhere from being selected.
        """
        result=image.copy()
        side=str(side).lower()
        for d in differences or []:
            if not isinstance(d,dict): continue
            status=str(d.get("type","replace")).lower()
            line_key="approval_line" if side=="approval" else "sample_line"
            line_index=d.get(line_key)
            if side=="approval":
                targets=d.get("approval",[]) or []
                color=self.ORANGE if status in {"delete","missing"} else self.RED
                label="MISSING" if status in {"delete","missing"} else "CHANGE"
            else:
                targets=d.get("sample",[]) or []
                color=self.BLUE if status in {"insert","extra"} else self.RED
                label="EXTRA" if status in {"insert","extra"} else "CHANGE"
            matches=self._find_on_line(words,targets,line_index)
            # If line index is unavailable, search phrase globally as a controlled fallback.
            if not matches and line_index is None:
                matches=self._find_phrase_match(words," ".join(targets))
            for item in matches:
                self._draw_box(result,self._safe_box(result,item,3),color,label,2)
        return self._save(result,output_path)

    def highlight_complete_label(self,image,words,mismatches=None,output_path=None,side="sample",visual_differences=None):
        if image is None: raise ValueError("Image cannot be None.")
        if output_path is None: output_path=self.create_output_path()
        diffs=visual_differences
        if diffs is None:
            diffs=[]
            for m in mismatches or []:
                diffs.extend(m.get("word_differences",[]) or m.get("differences",[]) or [])
        return self.highlight_visual_differences(image,words,diffs,side,output_path)

    def create_side_by_side(self, approval_image, sample_image, output_path, title="APPROVAL vs SAMPLE"):
        if approval_image is None or sample_image is None:
            raise ValueError("Both Approval and Sample images are required.")
        left=approval_image.copy(); right=sample_image.copy()
        target_h=max(left.shape[0],right.shape[0])
        def rh(img,h):
            if img.shape[0]==h:return img
            scale=h/img.shape[0]
            return cv2.resize(img,(max(1,int(img.shape[1]*scale)),h),interpolation=cv2.INTER_AREA)
        left,right=rh(left,target_h),rh(right,target_h)
        header=np.zeros((72,left.shape[1]+right.shape[1],3),dtype=np.uint8)
        cv2.putText(header,title,(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.0,self.WHITE,2,cv2.LINE_AA)
        # Divider and labels
        canvas=cv2.hconcat([left,right])
        mid=left.shape[1]
        cv2.line(canvas,(mid,0),(mid,target_h),(255,255,255),2)
        cv2.rectangle(canvas,(10,10),(160,42),(0,0,0),-1)
        cv2.putText(canvas,"APPROVAL",(18,34),cv2.FONT_HERSHEY_SIMPLEX,.65,self.WHITE,2,cv2.LINE_AA)
        cv2.rectangle(canvas,(mid+10,10),(mid+145,42),(0,0,0),-1)
        cv2.putText(canvas,"SAMPLE",(mid+18,34),cv2.FONT_HERSHEY_SIMPLEX,.65,self.WHITE,2,cv2.LINE_AA)
        return self._save(cv2.vconcat([header,canvas]),output_path)
