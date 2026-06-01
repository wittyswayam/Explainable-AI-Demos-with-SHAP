{{/*
XAI Platform — Helm Template Helpers
=====================================
Referenced by all templates in this chart.
Without this file, `helm template` and `helm install` both fail with
"function not defined" errors on every include call.
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "xai-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec). If the release name contains the chart name it will
be used as a full name.
*/}}
{{- define "xai-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label value: <chart-name>-<chart-version>
Used in the "helm.sh/chart" label.
*/}}
{{- define "xai-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to all resources managed by this chart.
Follows Kubernetes recommended label conventions:
  https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/
*/}}
{{- define "xai-platform.labels" -}}
helm.sh/chart: {{ include "xai-platform.chart" . }}
{{ include "xai-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: xai-platform
app.kubernetes.io/component: api
{{- end }}

{{/*
Selector labels — used in both Deployment.spec.selector and Service.spec.selector.
Must be stable across upgrades (do NOT include helm.sh/chart which changes on version bump).
*/}}
{{- define "xai-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "xai-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use.
If serviceAccount.create=true, create one named after the chart fullname;
otherwise use the explicitly specified name, or "default".
*/}}
{{- define "xai-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "xai-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the image pull policy.
Defaults to "IfNotPresent" for explicit tags, "Always" for "latest".
*/}}
{{- define "xai-platform.imagePullPolicy" -}}
{{- if eq .Values.image.tag "latest" }}
{{- "Always" }}
{{- else }}
{{- .Values.image.pullPolicy | default "IfNotPresent" }}
{{- end }}
{{- end }}

{{/*
Return the full image reference: repository:tag
*/}}
{{- define "xai-platform.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}

{{/*
Render environment variables from .Values.env map.
Usage: {{ include "xai-platform.envVars" . | nindent 12 }}
*/}}
{{- define "xai-platform.envVars" -}}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}

{{/*
Render secret environment variables from .Values.secretRef.
Assumes the Secret has keys: redis-url, secret-key, api-username, api-password.
*/}}
{{- define "xai-platform.secretEnvVars" -}}
- name: REDIS_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretRef }}
      key: redis-url
- name: SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretRef }}
      key: secret-key
{{- if .Values.auth }}
{{- if .Values.auth.enabled }}
- name: API_USERNAME
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretRef }}
      key: api-username
      optional: true
- name: API_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.secretRef }}
      key: api-password
      optional: true
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate a standard set of annotations for Prometheus scraping.
Usage: {{ include "xai-platform.prometheusAnnotations" . | nindent 8 }}
*/}}
{{- define "xai-platform.prometheusAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/path: "/metrics"
prometheus.io/port: "8000"
prometheus.io/scheme: "http"
{{- end }}

{{/*
Render topology spread constraints block from values.
Returns empty string if topologySpreadConstraints is not set.
*/}}
{{- define "xai-platform.topologySpreadConstraints" -}}
{{- with .Values.topologySpreadConstraints }}
topologySpreadConstraints:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Validate required values and emit a friendly error message if missing.
Call at the top of any template that depends on critical values.

Usage (in deployment.yaml):
  {{- include "xai-platform.validateValues" . }}
*/}}
{{- define "xai-platform.validateValues" -}}
{{- if not .Values.secretRef }}
{{- fail "values.secretRef is required. Create a Kubernetes Secret with keys: redis-url, secret-key" }}
{{- end }}
{{- if not .Values.image.repository }}
{{- fail "values.image.repository is required." }}
{{- end }}
{{- if and .Values.ingress.enabled (not .Values.ingress.hosts) }}
{{- fail "values.ingress.hosts must be set when ingress.enabled=true" }}
{{- end }}
{{- end }}

{{/*
HPA target reference block — reused in hpa.yaml template.
*/}}
{{- define "xai-platform.hpaTarget" -}}
scaleTargetRef:
  apiVersion: apps/v1
  kind: Deployment
  name: {{ include "xai-platform.fullname" . }}
{{- end }}

{{/*
Return "true" if TLS is enabled for any ingress host.
*/}}
{{- define "xai-platform.tlsEnabled" -}}
{{- if and .Values.ingress.enabled .Values.ingress.tls }}
{{- "true" }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}

{{/*
Resource limits block with sensible defaults if not set.
*/}}
{{- define "xai-platform.resources" -}}
{{- if .Values.resources }}
resources:
  {{- toYaml .Values.resources | nindent 2 }}
{{- else }}
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
{{- end }}
{{- end }}
