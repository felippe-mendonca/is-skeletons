# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: options.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='options.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\roptions.proto\"\xff\x02\n\x07Options\x12\x12\n\nbroker_uri\x18\x01 \x01(\t\x12\x13\n\x0bzipkin_host\x18\x02 \x01(\t\x12\x13\n\x0bzipkin_port\x18\x03 \x01(\r\x12\x19\n\x11zipkin_batch_size\x18\t \x01(\r\x12\x15\n\rmodels_folder\x18\x04 \x01(\t\x12\x1d\n\x05model\x18\x05 \x01(\x0e\x32\x0e.Options.Model\x12\x1f\n\x06resize\x18\x06 \x01(\x0b\x32\x0f.Options.Resize\x12\x18\n\x10resize_out_ratio\x18\x07 \x01(\x01\x12\x14\n\x0crender_topic\x18\x08 \x01(\r\x12\x1c\n\x14gpu_mem_allow_growth\x18\n \x01(\x08\x12\'\n\x1fper_process_gpu_memory_fraction\x18\x0b \x01(\x01\x1a\'\n\x06Resize\x12\r\n\x05width\x18\x01 \x01(\r\x12\x0e\n\x06height\x18\x02 \x01(\r\"$\n\x05Model\x12\x07\n\x03\x43MU\x10\x00\x12\x12\n\x0eMOBILENET_THIN\x10\x01\x62\x06proto3')
)



_OPTIONS_MODEL = _descriptor.EnumDescriptor(
  name='Model',
  full_name='Options.Model',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CMU', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MOBILENET_THIN', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=365,
  serialized_end=401,
)
_sym_db.RegisterEnumDescriptor(_OPTIONS_MODEL)


_OPTIONS_RESIZE = _descriptor.Descriptor(
  name='Resize',
  full_name='Options.Resize',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='Options.Resize.width', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='Options.Resize.height', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=324,
  serialized_end=363,
)

_OPTIONS = _descriptor.Descriptor(
  name='Options',
  full_name='Options',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='broker_uri', full_name='Options.broker_uri', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zipkin_host', full_name='Options.zipkin_host', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zipkin_port', full_name='Options.zipkin_port', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='zipkin_batch_size', full_name='Options.zipkin_batch_size', index=3,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='models_folder', full_name='Options.models_folder', index=4,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='Options.model', index=5,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize', full_name='Options.resize', index=6,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_out_ratio', full_name='Options.resize_out_ratio', index=7,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='render_topic', full_name='Options.render_topic', index=8,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gpu_mem_allow_growth', full_name='Options.gpu_mem_allow_growth', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='per_process_gpu_memory_fraction', full_name='Options.per_process_gpu_memory_fraction', index=10,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_OPTIONS_RESIZE, ],
  enum_types=[
    _OPTIONS_MODEL,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=401,
)

_OPTIONS_RESIZE.containing_type = _OPTIONS
_OPTIONS.fields_by_name['model'].enum_type = _OPTIONS_MODEL
_OPTIONS.fields_by_name['resize'].message_type = _OPTIONS_RESIZE
_OPTIONS_MODEL.containing_type = _OPTIONS
DESCRIPTOR.message_types_by_name['Options'] = _OPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Options = _reflection.GeneratedProtocolMessageType('Options', (_message.Message,), dict(

  Resize = _reflection.GeneratedProtocolMessageType('Resize', (_message.Message,), dict(
    DESCRIPTOR = _OPTIONS_RESIZE,
    __module__ = 'options_pb2'
    # @@protoc_insertion_point(class_scope:Options.Resize)
    ))
  ,
  DESCRIPTOR = _OPTIONS,
  __module__ = 'options_pb2'
  # @@protoc_insertion_point(class_scope:Options)
  ))
_sym_db.RegisterMessage(Options)
_sym_db.RegisterMessage(Options.Resize)


# @@protoc_insertion_point(module_scope)
