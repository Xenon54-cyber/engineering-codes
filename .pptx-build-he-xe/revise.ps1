$ErrorActionPreference = 'Stop'

$aNs = 'http://schemas.openxmlformats.org/drawingml/2006/main'
$pNs = 'http://schemas.openxmlformats.org/presentationml/2006/main'
$rNs = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

function Get-Manager([xml]$doc) {
  $manager = [System.Xml.XmlNamespaceManager]::new($doc.NameTable)
  [void]$manager.AddNamespace('a', $aNs)
  [void]$manager.AddNamespace('p', $pNs)
  [void]$manager.AddNamespace('r', $rNs)
  Write-Output -NoEnumerate $manager
}

function Set-Title([string]$path, [string]$match, [string]$value) {
  [xml]$doc = Get-Content -Raw -LiteralPath $path
  [System.Xml.XmlNamespaceManager]$manager = Get-Manager $doc
  $shape = @($doc.DocumentElement.SelectNodes('//p:sp', $manager) | Where-Object { $_.InnerText -like "*$match*" })[0]
  if ($null -eq $shape) { throw "Title containing '$match' was not found in $path" }
  $body = $shape.SelectSingleNode('./p:txBody', $manager)
  $sample = $body.SelectSingleNode('.//a:rPr', $manager)
  $body.RemoveAll()
  $bodyPr = $doc.CreateElement('a', 'bodyPr', $aNs)
  $list = $doc.CreateElement('a', 'lstStyle', $aNs)
  $para = $doc.CreateElement('a', 'p', $aNs)
  $run = $doc.CreateElement('a', 'r', $aNs)
  $rPr = if ($sample) { $doc.ImportNode($sample, $true) } else { $doc.CreateElement('a', 'rPr', $aNs) }
  $run.AppendChild($rPr) | Out-Null
  $text = $doc.CreateElement('a', 't', $aNs); $text.InnerText = $value
  $run.AppendChild($text) | Out-Null
  $para.AppendChild($run) | Out-Null
  $end = if ($sample) { $doc.ImportNode($sample, $true) } else { $doc.CreateElement('a', 'endParaRPr', $aNs) }
  if ($end.LocalName -eq 'rPr') { $end = $doc.CreateElement('a', 'endParaRPr', $aNs); $end.SetAttribute('lang','ru-RU') }
  $para.AppendChild($end) | Out-Null
  $body.AppendChild($bodyPr) | Out-Null; $body.AppendChild($list) | Out-Null; $body.AppendChild($para) | Out-Null
  $doc.Save($path)
}

function Add-TextBox([string]$path, [string[]]$lines, [int]$x, [int]$y, [int]$cx, [int]$cy, [int]$size = 1800) {
  [xml]$doc = Get-Content -Raw -LiteralPath $path
  [System.Xml.XmlNamespaceManager]$manager = Get-Manager $doc
  $tree = $doc.DocumentElement.SelectSingleNode('//p:spTree', $manager)
  $ids = @($doc.DocumentElement.SelectNodes('//p:cNvPr/@id', $manager) | ForEach-Object { [int]$_.Value })
  $id = (($ids | Measure-Object -Maximum).Maximum + 1)
  $shape = $doc.CreateElement('p', 'sp', $pNs)
  $nv = $doc.CreateElement('p', 'nvSpPr', $pNs)
  $cnv = $doc.CreateElement('p', 'cNvPr', $pNs); $cnv.SetAttribute('id', $id); $cnv.SetAttribute('name', "Вывод $id")
  $cnvsp = $doc.CreateElement('p', 'cNvSpPr', $pNs); $cnvsp.SetAttribute('txBox', '1')
  $nvPr = $doc.CreateElement('p', 'nvPr', $pNs)
  $nv.AppendChild($cnv) | Out-Null; $nv.AppendChild($cnvsp) | Out-Null; $nv.AppendChild($nvPr) | Out-Null
  $spPr = $doc.CreateElement('p', 'spPr', $pNs)
  $xfrm = $doc.CreateElement('a', 'xfrm', $aNs)
  $off = $doc.CreateElement('a', 'off', $aNs); $off.SetAttribute('x', $x); $off.SetAttribute('y', $y)
  $ext = $doc.CreateElement('a', 'ext', $aNs); $ext.SetAttribute('cx', $cx); $ext.SetAttribute('cy', $cy)
  $xfrm.AppendChild($off) | Out-Null; $xfrm.AppendChild($ext) | Out-Null
  $geom = $doc.CreateElement('a', 'prstGeom', $aNs); $geom.SetAttribute('prst', 'rect'); $geom.AppendChild($doc.CreateElement('a','avLst',$aNs)) | Out-Null
  $spPr.AppendChild($xfrm) | Out-Null; $spPr.AppendChild($geom) | Out-Null; $spPr.AppendChild($doc.CreateElement('a','noFill',$aNs)) | Out-Null
  $body = $doc.CreateElement('p', 'txBody', $pNs)
  $bodyPr = $doc.CreateElement('a','bodyPr',$aNs); $bodyPr.SetAttribute('wrap','square'); $bodyPr.SetAttribute('lIns','0'); $bodyPr.SetAttribute('rIns','0')
  $body.AppendChild($bodyPr) | Out-Null; $body.AppendChild($doc.CreateElement('a','lstStyle',$aNs)) | Out-Null
  foreach ($line in $lines) {
    $para = $doc.CreateElement('a','p',$aNs)
    $pPr = $doc.CreateElement('a','pPr',$aNs); $pPr.SetAttribute('algn','l'); $pPr.SetAttribute('spcBef','0')
    $para.AppendChild($pPr) | Out-Null
    $run = $doc.CreateElement('a','r',$aNs)
    $rPr = $doc.CreateElement('a','rPr',$aNs); $rPr.SetAttribute('lang','ru-RU'); $rPr.SetAttribute('sz',$size); $rPr.SetAttribute('dirty','0')
    $latin = $doc.CreateElement('a','latin',$aNs); $latin.SetAttribute('typeface','Calibri')
    $cs = $doc.CreateElement('a','cs',$aNs); $cs.SetAttribute('typeface','Calibri')
    $rPr.AppendChild($latin) | Out-Null; $rPr.AppendChild($cs) | Out-Null
    $run.AppendChild($rPr) | Out-Null
    $t = $doc.CreateElement('a','t',$aNs); $t.InnerText = $line; $run.AppendChild($t) | Out-Null
    $para.AppendChild($run) | Out-Null
    $end = $doc.CreateElement('a','endParaRPr',$aNs); $end.SetAttribute('lang','ru-RU'); $end.SetAttribute('sz',$size); $end.AppendChild($doc.ImportNode($latin,$true)) | Out-Null; $end.AppendChild($doc.ImportNode($cs,$true)) | Out-Null
    $para.AppendChild($end) | Out-Null; $body.AppendChild($para) | Out-Null
  }
  $shape.AppendChild($nv) | Out-Null; $shape.AppendChild($spPr) | Out-Null; $shape.AppendChild($body) | Out-Null
  $tree.AppendChild($shape) | Out-Null
  $doc.Save($path)
}

Set-Title 'unpacked\ppt\slides\slide8.xml' 'Результаты оптимизации' 'Оптимум He–Xe цикла по температуре на входе в реактор'
Add-TextBox 'unpacked\ppt\slides\slide8.xml' @(
  'Вывод. Максимальный КПД нетто 45,871 % получен при P₁ = 5,20 МПа и P₂ = 2,75 МПа.',
  'При T₆ = 750 °C одновременно достигаются максимум КПД и мощности нетто. Дальнейшее повышение температуры снижает оба показателя.'
) 371475 4510000 10732705 900000 1650

Set-Title 'unpacked\ppt\slides\slide9.xml' 'Результаты оптимизации' 'Степень расширения и КПД нетто He–Xe цикла'
Add-TextBox 'unpacked\ppt\slides\slide9.xml' @(
  'Вывод. При T₆ = 750 °C максимум КПД достигается при P₁/P₂ ≈ 1,89. При снижении температуры требуется большая степень расширения, но достижимый КПД уменьшается.'
) 371475 4720000 10732705 560000 1500

Set-Title 'unpacked\ppt\slides\slide10.xml' 'Результаты анализа чувствительности' 'Чувствительность КПД нетто к параметрам оборудования'

[xml]$summary = Get-Content -Raw 'unpacked\ppt\slides\slide11.xml'
[System.Xml.XmlNamespaceManager]$mgr = Get-Manager $summary
$tree = $summary.DocumentElement.SelectSingleNode('//p:spTree', $mgr)
$tree.RemoveAll()
$grp = $summary.CreateElement('p','nvGrpSpPr',$pNs); $tree.AppendChild($grp) | Out-Null
$cnv = $summary.CreateElement('p','cNvPr',$pNs); $cnv.SetAttribute('id','1'); $cnv.SetAttribute('name',''); $grp.AppendChild($cnv) | Out-Null
$grp.AppendChild($summary.CreateElement('p','cNvGrpSpPr',$pNs)) | Out-Null; $grp.AppendChild($summary.CreateElement('p','nvPr',$pNs)) | Out-Null
$grpPr = $summary.CreateElement('p','grpSpPr',$pNs); $xfrm = $summary.CreateElement('a','xfrm',$aNs); foreach($item in @(@('off',0,0),@('ext',0,0),@('chOff',0,0),@('chExt',0,0))){$e=$summary.CreateElement('a',$item[0],$aNs); if($item[0] -in 'off','chOff'){$e.SetAttribute('x',$item[1]);$e.SetAttribute('y',$item[2])}else{$e.SetAttribute('cx',$item[1]);$e.SetAttribute('cy',$item[2])};$xfrm.AppendChild($e)|Out-Null}; $grpPr.AppendChild($xfrm)|Out-Null; $tree.AppendChild($grpPr)|Out-Null
$summary.Save('unpacked\ppt\slides\slide11.xml')
Add-TextBox 'unpacked\ppt\slides\slide11.xml' @('Ключевые выводы оптимизации') 371475 120000 10732705 500000 2400
Add-TextBox 'unpacked\ppt\slides\slide11.xml' @(
  '1. Рабочий режим He–Xe цикла: P₁ = 5,20 МПа, P₂ = 2,75 МПа, T₆ = 750 °C; КПД нетто 45,871 %.',
  '2. Минимальный температурный напор рекуператора 1,21 K дает максимум КПД 46,606 % и требует проверки технической реализуемости.',
  '3. КПД наиболее чувствителен к КПД турбины и компрессора. При ηТ = 0,927 и ηК = 0,930 расчетный максимум составляет 48,297 %.',
  '4. Основное снижение полезной мощности связано с отводом теплоты в охладителе: 1,095 МВт при мощности нетто 1,261 МВт.'
) 650000 1050000 10100000 3300000 1850

# Exclude the empty divider slide and the duplicate flow-sheet slide from presentation order.
[xml]$presentation = Get-Content -Raw 'unpacked\ppt\presentation.xml'
[System.Xml.XmlNamespaceManager]$pm = Get-Manager $presentation
$rels = Get-Content -Raw 'unpacked\ppt\_rels\presentation.xml.rels'
foreach ($id in @('rId5','rId18')) {
  $node = $presentation.DocumentElement.SelectSingleNode("//p:sldId[@r:id='$id']", $pm)
  if ($node) { $node.ParentNode.RemoveChild($node) | Out-Null }
}
$presentation.Save('unpacked\ppt\presentation.xml')

$dest = Resolve-Path 'unpacked'
$final = Join-Path (Resolve-Path '..') 'Результаты He-Xe — оформлено.pptx'
if (Test-Path -LiteralPath $final) { Remove-Item -LiteralPath $final -Force }
Compress-Archive -Path (Join-Path $dest '*') -DestinationPath $final -CompressionLevel Optimal
Write-Output $final
