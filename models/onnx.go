package models

import (
	"fmt"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// ONNXModel wraps an ONNX Runtime session for inference
type ONNXModel struct {
	session      *onnxruntime.DynamicAdvancedSession
	inputNames   []string
	outputNames  []string
	inputShapes  []onnxruntime.Shape
	inputTypes   []onnxruntime.TensorElementDataType
	outputShapes []onnxruntime.Shape
	outputTypes  []onnxruntime.TensorElementDataType
}

// NewONNXModel creates a new ONNX model from a file
func NewONNXModel(modelPath string) (*ONNXModel, error) {
	// Initialize ONNX Runtime (once per process)
	err := onnxruntime.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Create session options
	options, err := onnxruntime.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() {
		_ = options.Destroy()
	}()

	// Try to use CUDA if available
	// Note: This requires ONNX Runtime GPU version
	// options.AppendExecutionProviderCUDA(0)

	// Get input/output information from the model file first
	inputs, outputs, err := onnxruntime.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get model input/output info: %w", err)
	}

	// Extract names, shapes, and types
	inputNames := make([]string, len(inputs))
	inputShapes := make([]onnxruntime.Shape, len(inputs))
	inputTypes := make([]onnxruntime.TensorElementDataType, len(inputs))
	for i, input := range inputs {
		inputNames[i] = input.Name
		inputShapes[i] = input.Dimensions
		inputTypes[i] = input.DataType
	}

	outputNames := make([]string, len(outputs))
	outputShapes := make([]onnxruntime.Shape, len(outputs))
	outputTypes := make([]onnxruntime.TensorElementDataType, len(outputs))
	for i, output := range outputs {
		outputNames[i] = output.Name
		outputShapes[i] = output.Dimensions
		outputTypes[i] = output.DataType
	}

	// Create dynamic session with input/output names
	session, err := onnxruntime.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ONNXModel{
		session:      session,
		inputNames:   inputNames,
		outputNames:  outputNames,
		inputShapes:  inputShapes,
		inputTypes:   inputTypes,
		outputShapes: outputShapes,
		outputTypes:  outputTypes,
	}, nil
}

// Run runs inference on the model
func (m *ONNXModel) Run(inputs map[string]interface{}) (map[string]interface{}, error) {
	// Create input tensors as Values
	inputValues := make([]onnxruntime.Value, len(m.inputNames))
	defer func() {
		for _, value := range inputValues {
			if value != nil {
				_ = value.Destroy()
			}
		}
	}()

	for i, name := range m.inputNames {
		input, exists := inputs[name]
		if !exists {
			return nil, fmt.Errorf("missing input: %s", name)
		}

		// Convert input to tensor
		tensor, err := m.createTensor(input)
		if err != nil {
			return nil, fmt.Errorf("failed to create tensor for %s: %w", name, err)
		}
		inputValues[i] = tensor
	}

	// Create output values slice (nil values will be allocated by Run)
	outputValues := make([]onnxruntime.Value, len(m.outputNames))
	defer func() {
		for _, value := range outputValues {
			if value != nil {
				_ = value.Destroy()
			}
		}
	}()

	// Run inference - DynamicAdvancedSession.Run takes inputs and outputs slices
	err := m.session.Run(inputValues, outputValues)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Extract outputs
	outputs := make(map[string]interface{})
	for i, name := range m.outputNames {
		if outputValues[i] != nil {
			// Try to cast to Tensor[float32] to get data
			if tensor, ok := outputValues[i].(*onnxruntime.Tensor[float32]); ok {
				data := tensor.GetData()
				shape := tensor.GetShape()
				outputs[name] = map[string]interface{}{
					"data":  data,
					"shape": shape,
				}
			} else {
				return nil, fmt.Errorf("unsupported output type for %s", name)
			}
		}
	}

	return outputs, nil
}

// createTensor creates an ONNX tensor from input data
func (m *ONNXModel) createTensor(input interface{}) (*onnxruntime.Tensor[float32], error) {
	switch v := input.(type) {
	case []float32:
		// 1D tensor
		shape := []int64{int64(len(v))}
		return onnxruntime.NewTensor(shape, v)
	case [][]float32:
		// 2D tensor
		rows := int64(len(v))
		cols := int64(0)
		if rows > 0 {
			cols = int64(len(v[0]))
		}
		shape := []int64{rows, cols}

		// Flatten
		flat := make([]float32, rows*cols)
		for i, row := range v {
			copy(flat[i*int(cols):(i+1)*int(cols)], row)
		}

		return onnxruntime.NewTensor(shape, flat)
	case [][][]float32:
		// 3D tensor
		dim0 := int64(len(v))
		dim1 := int64(0)
		dim2 := int64(0)
		if dim0 > 0 {
			dim1 = int64(len(v[0]))
			if dim1 > 0 {
				dim2 = int64(len(v[0][0]))
			}
		}
		shape := []int64{dim0, dim1, dim2}

		// Flatten
		flat := make([]float32, dim0*dim1*dim2)
		idx := 0
		for _, mat := range v {
			for _, row := range mat {
				copy(flat[idx:idx+int(dim2)], row)
				idx += int(dim2)
			}
		}

		return onnxruntime.NewTensor(shape, flat)
	case []int64:
		// Integer tensor (for token IDs)
		shape := []int64{int64(len(v))}
		// Convert to float32 for now (ONNX Runtime Go binding limitation)
		floatData := make([]float32, len(v))
		for i, val := range v {
			floatData[i] = float32(val)
		}
		return onnxruntime.NewTensor(shape, floatData)
	case [][]int64:
		// 2D integer tensor
		rows := int64(len(v))
		cols := int64(0)
		if rows > 0 {
			cols = int64(len(v[0]))
		}
		shape := []int64{rows, cols}

		// Flatten and convert to float32
		flat := make([]float32, rows*cols)
		for i, row := range v {
			for j, val := range row {
				flat[i*int(cols)+j] = float32(val)
			}
		}

		return onnxruntime.NewTensor(shape, flat)
	default:
		return nil, fmt.Errorf("unsupported input type: %T", input)
	}
}

// Close releases model resources
func (m *ONNXModel) Close() error {
	if m.session != nil {
		err := m.session.Destroy()
		m.session = nil
		if err != nil {
			return fmt.Errorf("failed to destroy session: %w", err)
		}
	}
	return nil
}

// GetInputNames returns the names of model inputs
func (m *ONNXModel) GetInputNames() []string {
	return m.inputNames
}

// GetOutputNames returns the names of model outputs
func (m *ONNXModel) GetOutputNames() []string {
	return m.outputNames
}

// GetInputShapes returns the shapes of model inputs
func (m *ONNXModel) GetInputShapes() []onnxruntime.Shape {
	return m.inputShapes
}

// GetOutputShapes returns the shapes of model outputs
func (m *ONNXModel) GetOutputShapes() []onnxruntime.Shape {
	return m.outputShapes
}

// GetInputTypes returns the data types of model inputs
func (m *ONNXModel) GetInputTypes() []onnxruntime.TensorElementDataType {
	return m.inputTypes
}

// GetOutputTypes returns the data types of model outputs
func (m *ONNXModel) GetOutputTypes() []onnxruntime.TensorElementDataType {
	return m.outputTypes
}
